package org.apache.spark.ml.tree.impl

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.tree.{ImpurityAggregatorSingle, ygg, Node => SparkNode}
import org.apache.spark.ml.tree.impl.Yggdrasil.{FeatureVector, PartitionInfo, YggdrasilMetadata}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.collection.BitSet
import org.roaringbitmap.RoaringBitmap

object YggdrasilRegression {

  def trainImpl(
                 input: RDD[LabeledPoint],
                 colStoreInit: RDD[(Int, Array[Double])],
                 metadata: YggdrasilMetadata,
                 numRows: Int,
                 maxDepth: Int): SparkNode = {

    val labels = new Array[Double](numRows)
    input.map(_.label).zipWithIndex().collect().foreach { case (label: Double, rowIndex: Long) =>
      labels(rowIndex.toInt) = label.toDouble
    }
    val labelsBc = input.sparkContext.broadcast(labels)
    // NOTE: Labels are not sorted with features since that would require 1 copy per feature,
    //       rather than 1 copy per worker. This means a lot of random accesses.
    //       We could improve this by applying first-level sorting (by node) to labels.

    // Sort each column by feature values.
    val colStore: RDD[FeatureVector] = colStoreInit.map { case (featureIndex, col) =>
      val featureArity: Int = metadata.categoricalFeaturesInfo.getOrElse(featureIndex, 0)
      FeatureVector.fromOriginal(featureIndex, featureArity, col)
    }
    // Group columns together into one array of columns per partition.
    // TODO: Test avoiding this grouping, and see if it matters.
    val groupedColStore: RDD[Array[FeatureVector]] = colStore.mapPartitions {
      iterator: Iterator[FeatureVector] =>
        if (iterator.nonEmpty) Iterator(iterator.toArray) else Iterator()
    }
    groupedColStore.persist(StorageLevel.MEMORY_AND_DISK)

    // Initialize partitions with 1 node (each instance at the root node).
    val fullImpurityAgg = metadata.createImpurityAggregator()
    var i = 0
    while (i < labels.length) {
      fullImpurityAgg.update(labels(i))
      i += 1
    }
    var partitionInfos: RDD[PartitionInfo] = groupedColStore.map { groupedCols =>
      val initActive = new BitSet(1)
      initActive.set(0)

      new PartitionInfo(groupedCols, Array[Int](0, numRows), initActive, Array(fullImpurityAgg))
    }

    // Initialize model.
    // Note: We do not use node indices.
    val rootNode = ygg.LearningNode.emptyNode(1) // TODO: remove node id
    // Active nodes (still being split), updated each iteration
    var activeNodePeriphery: Array[ygg.LearningNode] = Array(rootNode)
    var numNodeOffsets: Int = 2

    // Iteratively learn, one level of the tree at a time.
    var currentLevel = 0
    var doneLearning = false
    while (currentLevel < maxDepth && !doneLearning) {
      // Compute best split for each active node.
      val bestSplitsAndGains: Array[(Option[ygg.Split], ImpurityStats)] =
        computeBestSplits(partitionInfos, labelsBc, metadata)
      /*
      // NOTE: The actual active nodes (activeNodePeriphery) may be a subset of the nodes under
      //       bestSplitsAndGains since
      assert(activeNodePeriphery.length == bestSplitsAndGains.length,
        s"activeNodePeriphery.length=${activeNodePeriphery.length} does not equal" +
          s" bestSplitsAndGains.length=${bestSplitsAndGains.length}")
      */

      // Update current model and node periphery.
      // Note: This flatMap has side effects (on the model).
      activeNodePeriphery =
        Yggdrasil.computeActiveNodePeriphery(activeNodePeriphery, bestSplitsAndGains, metadata.minInfoGain)
      // We keep all old nodeOffsets and add one for each node split.
      // Each node split adds 2 nodes to activeNodePeriphery.
      // TODO: Should this be calculated after filtering for impurity??
      numNodeOffsets = numNodeOffsets + activeNodePeriphery.length / 2

      // Filter active node periphery by impurity.
      val estimatedRemainingActive = activeNodePeriphery.count(_.stats.impurity > 0.0)

      // TODO: Check to make sure we split something, and stop otherwise.
      doneLearning = currentLevel + 1 >= maxDepth || estimatedRemainingActive == 0

      if (!doneLearning) {
        val splits: Array[Option[ygg.Split]] = bestSplitsAndGains.map(_._1)

        // Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right
        val aggBitVector: RoaringBitmap = Yggdrasil.aggregateBitVector(partitionInfos, splits, numRows)
        val newPartitionInfos = partitionInfos.map { partitionInfo =>
          val bv = new BitSet(numRows)
          val iter = aggBitVector.getIntIterator
          while(iter.hasNext) {
            bv.set(iter.next)
          }
          partitionInfo.update(bv, numNodeOffsets, labelsBc.value, metadata)
        }
        // TODO: remove.  For some reason, this is needed to make things work.
        // Probably messing up somewhere above...
        newPartitionInfos.cache().count()
        partitionInfos = newPartitionInfos

      }
      currentLevel += 1
    }

    // Done with learning
    groupedColStore.unpersist()
    labelsBc.unpersist()
    rootNode.toSparkNode
  }

  /**
   * Find the best splits for all active nodes.
   *  - On each partition, for each feature on the partition, select the best split for each node.
   *    Each worker returns: For each active node, best split + info gain
   *  - The splits across workers are aggregated to the driver.
   * @return  Array over active nodes of (best split, impurity stats for split),
   *          where the split is None if no useful split exists
   */
  private[impl] def computeBestSplits(
                                       partitionInfos: RDD[PartitionInfo],
                                       labelsBc: Broadcast[Array[Double]],
                                       metadata: YggdrasilMetadata) = {
    // On each partition, for each feature on the partition, select the best split for each node.
    // This will use:
    //  - groupedColStore (the features)
    //  - partitionInfos (the node -> instance mapping)
    //  - labelsBc (the labels column)
    // Each worker returns:
    //   for each active node, best split + info gain,
    //     where the best split is None if no useful split exists
    val partBestSplitsAndGains: RDD[Array[(Option[ygg.Split], ImpurityStats)]] = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int],
      activeNodes: BitSet, fullImpurityAggs: Array[ImpurityAggregatorSingle]) =>
        val localLabels = labelsBc.value
        // Iterate over the active nodes in the current level.
        val toReturn = new Array[(Option[ygg.Split], ImpurityStats)](activeNodes.cardinality())
        val iter: Iterator[Int] = activeNodes.iterator
        var i = 0
        while (iter.hasNext) {
          val nodeIndexInLevel = iter.next
          val fromOffset = nodeOffsets(nodeIndexInLevel)
          val toOffset = nodeOffsets(nodeIndexInLevel + 1)
          val fullImpurityAgg = fullImpurityAggs(nodeIndexInLevel)
          val splitsAndStats =
            columns.map { col =>
              chooseSplit(col, localLabels, fromOffset, toOffset, fullImpurityAgg, metadata)
            }
          toReturn(i) = splitsAndStats.maxBy(_._2.gain)
          i += 1
        }
        toReturn
    }

    // Aggregate best split for each active node.
    partBestSplitsAndGains.treeReduce { case (splitsGains1, splitsGains2) =>
      splitsGains1.zip(splitsGains2).map { case ((split1, gain1), (split2, gain2)) =>
        if (gain1.gain >= gain2.gain) {
          (split1, gain1)
        } else {
          (split2, gain2)
        }
      }
    }
  }

  /**
   * Choose the best split for a feature at a node.
   * TODO: Return null or None when the split is invalid, such as putting all instances on one
   *       child node.
   *
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.
   */
  private[impl] def chooseSplit(
                                 col: FeatureVector,
                                 labels: Array[Double],
                                 fromOffset: Int,
                                 toOffset: Int,
                                 fullImpurityAgg: ImpurityAggregatorSingle,
                                 metadata: YggdrasilMetadata): (Option[ygg.Split], ImpurityStats) = {
    if (col.isCategorical) {
      if (metadata.isUnorderedFeature(col.featureIndex)) {
        val splits: Array[ygg.CategoricalSplit] = metadata.getUnorderedSplits(col.featureIndex)
        chooseUnorderedCategoricalSplit(col.featureIndex, col.values, col.indices, labels, fromOffset, toOffset,
          metadata, col.featureArity, splits)
      } else {
        chooseOrderedCategoricalSplit(col.featureIndex, col.values, col.indices, labels, fromOffset, toOffset,
          metadata, col.featureArity)
      }
    } else {
      chooseContinuousSplit(col.featureIndex, col.values, col.indices, labels, fromOffset, toOffset,
        fullImpurityAgg, metadata)
    }
  }

  /**
   * Find the best split for an ordered categorical feature at a single node.
   *
   * Algorithm:
   *  - For each category, compute a "centroid."
   *     - For multiclass classification, the centroid is the label impurity.
   *     - For binary classification and regression, the centroid is the average label.
   *  - Sort the centroids, and consider splits anywhere in this order.
   *    Thus, with K categories, we consider K - 1 possible splits.
   *
   * @param featureIndex  Index of feature being split.
   * @param values  Feature values at this node.  Sorted in increasing order.
   * @param labels  Labels corresponding to values, in the same order.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseOrderedCategoricalSplit(
                                                   featureIndex: Int,
                                                   values: Array[Double],
                                                   indices: Array[Int],
                                                   labels: Array[Double],
                                                   from: Int,
                                                   to: Int,
                                                   metadata: YggdrasilMetadata,
                                                   featureArity: Int): (Option[ygg.Split], ImpurityStats) = {
    // TODO: Support high-arity features by using a single array to hold the stats.

    // aggStats(category) = label statistics for category
    val aggStats = Array.tabulate[ImpurityAggregatorSingle](featureArity)(
      _ => metadata.createImpurityAggregator())
    var i = from
    while (i < to) {
      val cat = values(i)
      val label = labels(indices(i))
      aggStats(cat.toInt).update(label)
      i += 1
    }

    // Compute centroids.  centroidsForCategories is a list: (category, centroid)
    val centroidsForCategories: Seq[(Int, Double)] = if (metadata.isMulticlass) {
      // For categorical variables in multiclass classification,
      // the bins are ordered by the impurity of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          categoryStats.getCalculator.calculate()
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    } else if (metadata.isClassification) { // binary classification
      // For categorical variables in binary classification,
      // the bins are ordered by the centroid of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          assert(categoryStats.stats.length == 2)
          (categoryStats.stats(1) - categoryStats.stats(0)) / categoryStats.getCount
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    } else { // regression
      // For categorical variables in regression,
      // the bins are ordered by the centroid of their corresponding labels.
      Range(0, featureArity).map { case featureValue =>
        val categoryStats = aggStats(featureValue)
        val centroid = if (categoryStats.getCount != 0) {
          categoryStats.getCalculator.predict
        } else {
          Double.MaxValue
        }
        (featureValue, centroid)
      }
    }

    val categoriesSortedByCentroid: List[Int] = centroidsForCategories.toList.sortBy(_._2).map(_._1)

    // Cumulative sums of bin statistics for left, right parts of split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = metadata.createImpurityAggregator()
    var j = 0
    val length = aggStats.length
    while (j < length) {
      rightImpurityAgg.add(aggStats(j))
      j += 1
    }

    var bestSplitIndex: Int = -1  // index into categoriesSortedByCentroid
    val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Double = 0.0
    var rightCount: Double = rightImpurityAgg.getCount
    val fullCount: Double = rightCount

    // Consider all splits. These only cover valid splits, with at least one category on each side.
    val numSplits = categoriesSortedByCentroid.length - 1
    var sortedCatIndex = 0
    while (sortedCatIndex < numSplits) {
      val cat = categoriesSortedByCentroid(sortedCatIndex)
      // Update left, right stats
      val catStats = aggStats(cat)
      leftImpurityAgg.add(catStats)
      rightImpurityAgg.subtract(catStats)
      leftCount += catStats.getCount
      rightCount -= catStats.getCount
      // Compute impurity
      val leftWeight = leftCount / fullCount
      val rightWeight = rightCount / fullCount
      val leftImpurity = leftImpurityAgg.getCalculator.calculate()
      val rightImpurity = rightImpurityAgg.getCalculator.calculate()
      val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
      if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
        bestSplitIndex = sortedCatIndex
        System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
        bestGain = gain
      }
      sortedCatIndex += 1
    }

    val categoriesForSplit =
      categoriesSortedByCentroid.slice(0, bestSplitIndex + 1).map(_.toDouble)
    val bestFeatureSplit =
      new ygg.CategoricalSplit(featureIndex, categoriesForSplit.toArray, featureArity)
    val fullImpurityAgg = leftImpurityAgg.deepCopy().add(rightImpurityAgg)
    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftImpurityAgg.getCalculator, bestRightImpurityAgg.getCalculator)

    if (bestSplitIndex == -1 || bestGain == 0.0) {
      (None, bestImpurityStats)
    } else {
      (Some(bestFeatureSplit), bestImpurityStats)
    }
  }

  /**
   * Find the best split for an unordered categorical feature at a single node.
   *
   * Algorithm:
   *  - Considers all possible subsets (exponentially many)
   *
   * @param featureIndex  Index of feature being split.
   * @param values  Feature values at this node.  Sorted in increasing order.
   * @param labels  Labels corresponding to values, in the same order.
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseUnorderedCategoricalSplit(
                                                     featureIndex: Int,
                                                     values: Array[Double],
                                                     indices: Array[Int],
                                                     labels: Array[Double],
                                                     from: Int,
                                                     to: Int,
                                                     metadata: YggdrasilMetadata,
                                                     featureArity: Int,
                                                     splits: Array[ygg.CategoricalSplit]): (Option[ygg.Split], ImpurityStats) = {

    // Label stats for each category
    val aggStats = Array.tabulate[ImpurityAggregatorSingle](featureArity)(
      _ => metadata.createImpurityAggregator())
    var i = from
    while (i < to) {
      val cat = values(i)
      val label = labels(indices(i))
      // NOTE: we assume the values for categorical features are Ints in [0,featureArity)
      aggStats(cat.toInt).update(label)
      i += 1
    }

    // Aggregated statistics for left part of split and entire split.
    val leftImpurityAgg = metadata.createImpurityAggregator()
    val fullImpurityAgg = metadata.createImpurityAggregator()
    aggStats.foreach(fullImpurityAgg.add)
    val fullImpurity = fullImpurityAgg.getCalculator.calculate()

    if (featureArity == 1) {
      // All instances go right
      val impurityStats = new ImpurityStats(0.0, fullImpurityAgg.getCalculator.calculate(),
        fullImpurityAgg.getCalculator, leftImpurityAgg.getCalculator,
        fullImpurityAgg.getCalculator)
      (None, impurityStats)
    } else {
      //  TODO: We currently add and remove the stats for all categories for each split.
      //  A better way to do it would be to consider splits in an order such that each iteration
      //  only requires addition/removal of a single category and a single add/subtract to
      //  leftCount and rightCount.
      //  TODO: Use more efficient encoding such as gray codes
      var bestSplit: Option[ygg.CategoricalSplit] = None
      val bestLeftImpurityAgg = leftImpurityAgg.deepCopy()
      var bestGain: Double = -1.0
      val fullCount: Double = to - from
      for (split <- splits) {
        // Update left, right impurity stats
        split.leftCategories.foreach(c => leftImpurityAgg.add(aggStats(c.toInt)))
        val rightImpurityAgg = fullImpurityAgg.deepCopy().subtract(leftImpurityAgg)
        val leftCount = leftImpurityAgg.getCount
        val rightCount = rightImpurityAgg.getCount
        // Compute impurity
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
          bestSplit = Some(split)
          System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
          bestGain = gain
        }
        // Reset left impurity stats
        leftImpurityAgg.clear()
      }

      val bestFeatureSplit = bestSplit match {
        case Some(split) => Some(
          new ygg.CategoricalSplit(featureIndex, split.leftCategories, featureArity))
        case None => None

      }
      val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
      val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity,
        fullImpurityAgg.getCalculator, bestLeftImpurityAgg.getCalculator,
        bestRightImpurityAgg.getCalculator)
      (bestFeatureSplit, bestImpurityStats)
    }
  }

  /**
   * Choose splitting rule: feature value <= threshold
   * @return  (best split, statistics for split)  If the best split actually puts all instances
   *          in one leaf node, then it will be set to None.  The impurity stats maybe still be
   *          useful, so they are returned.
   */
  private[impl] def chooseContinuousSplit(
                                           featureIndex: Int,
                                           values: Array[Double],
                                           indices: Array[Int],
                                           labels: Array[Double],
                                           from: Int,
                                           to: Int,
                                           fullImpurityAgg: ImpurityAggregatorSingle,
                                           metadata: YggdrasilMetadata): (Option[ygg.Split], ImpurityStats) = {

    val leftImpurityAgg = metadata.createImpurityAggregator()
    val rightImpurityAgg = fullImpurityAgg.deepCopy()

    var bestThreshold: Double = Double.NegativeInfinity
    val bestLeftImpurityAgg = metadata.createImpurityAggregator()
    var bestGain: Double = 0.0
    val fullImpurity = rightImpurityAgg.getCalculator.calculate()
    var leftCount: Int = 0
    var rightCount: Int = to - from
    val fullCount: Double = rightCount
    var currentThreshold = values.headOption.getOrElse(bestThreshold)
    var j = from
    while (j < to) {
      val value = values(j)
      val label = labels(indices(j))
      if (value != currentThreshold) {
        // Check gain
        val leftWeight = leftCount / fullCount
        val rightWeight = rightCount / fullCount
        val leftImpurity = leftImpurityAgg.getCalculator.calculate()
        val rightImpurity = rightImpurityAgg.getCalculator.calculate()
        val gain = fullImpurity - leftWeight * leftImpurity - rightWeight * rightImpurity
        if (leftCount != 0 && rightCount != 0 && gain > bestGain && gain > metadata.minInfoGain) {
          bestThreshold = currentThreshold
          System.arraycopy(leftImpurityAgg.stats, 0, bestLeftImpurityAgg.stats, 0, leftImpurityAgg.stats.length)
          bestGain = gain
        }
        currentThreshold = value
      }
      // Move this instance from right to left side of split.
      leftImpurityAgg.update(label, 1)
      rightImpurityAgg.update(label, -1)
      leftCount += 1
      rightCount -= 1
      j += 1
    }

    val bestRightImpurityAgg = fullImpurityAgg.deepCopy().subtract(bestLeftImpurityAgg)
    val bestImpurityStats = new ImpurityStats(bestGain, fullImpurity, fullImpurityAgg.getCalculator,
      bestLeftImpurityAgg.getCalculator, bestRightImpurityAgg.getCalculator)
    val split = if (bestThreshold != Double.NegativeInfinity && bestThreshold != values.last) {
      Some(new ygg.ContinuousSplit(featureIndex, bestThreshold))
    } else {
      None
    }
    (split, bestImpurityStats)
  }
}
