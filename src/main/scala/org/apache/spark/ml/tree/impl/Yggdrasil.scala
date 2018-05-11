/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree.impl

import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.tree.impl.YggdrasilUtil._
import org.apache.spark.ml.tree.{ygg, Node => SparkNode, _}
import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, Strategy => OldStrategy}
import org.apache.spark.mllib.tree.impl.DecisionTreeMetadata
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.mllib.tree.model.{ImpurityStats, Predict}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.util.collection.{BitSet, SortDataFormat, Sorter}
import org.roaringbitmap.RoaringBitmap

final class YggdrasilClassifier(override val uid: String)
  extends Predictor[Vector, YggdrasilClassifier, DecisionTreeClassificationModel]
  with DecisionTreeParams with TreeClassifierParams {

  def this() = this(Identifiable.randomUID("yggc"))

  // Override parameter setters from parent trait for Java API compatibility.
  override def setMaxDepth(value: Int): this.type = super.setMaxDepth(value)

  override def setMaxBins(value: Int): this.type = super.setMaxBins(value)

  override def setMinInstancesPerNode(value: Int): this.type =
    super.setMinInstancesPerNode(value)

  override def setMinInfoGain(value: Double): this.type = super.setMinInfoGain(value)

  override def setMaxMemoryInMB(value: Int): this.type = super.setMaxMemoryInMB(value)

  override def setCacheNodeIds(value: Boolean): this.type = super.setCacheNodeIds(value)

  override def setCheckpointInterval(value: Int): this.type = super.setCheckpointInterval(value)

  override def setImpurity(value: String): this.type = super.setImpurity(value)

  override def setSeed(value: Long): this.type = super.setSeed(value)

  override def copy(extra: ParamMap): YggdrasilClassifier = defaultCopy(extra)

  def train(
             input: RDD[LabeledPoint],
             transposedDataset: RDD[(Int, Array[Double])],
             categoricalFeatures: Map[Int, Int]): DecisionTreeClassificationModel = {

    val numClasses: Int = input.map(_.label).distinct().count().toInt
    val strategy = getOldStrategy(categoricalFeatures, numClasses)
    val model = Yggdrasil.train(input, strategy, Some(transposedDataset), parentUID = Some(uid))
    model.asInstanceOf[DecisionTreeClassificationModel]
  }

  override protected def train(dataset: DataFrame): DecisionTreeClassificationModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = MetadataUtils.getNumClasses(dataset.schema($(labelCol))) match {
      case Some(n: Int) => n
      case None => throw new IllegalArgumentException("DecisionTreeClassifier was given input" +
        s" with invalid label column ${$(labelCol)}, without the number of classes" +
        " specified. See StringIndexer.")
      // TODO: Automatically index labels: SPARK-7126
    }

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    val strategy = getOldStrategy(categoricalFeatures, numClasses)
    val model = Yggdrasil.train(oldDataset, strategy, colStoreInput = None, parentUID = Some(uid))
    model.asInstanceOf[DecisionTreeClassificationModel]
  }

  /** Create a Strategy instance to use with the old API. */
  private[impl] def getOldStrategy(categoricalFeatures: Map[Int, Int], numClasses: Int): OldStrategy = {
    super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity,
      subsamplingRate = 1.0)
  }
}

final class YggdrasilRegressor(override val uid: String)
  extends Predictor[Vector, YggdrasilRegressor, DecisionTreeRegressionModel]
  with DecisionTreeParams with TreeRegressorParams {

  def this() = this(Identifiable.randomUID("yggr"))

  // Override parameter setters from parent trait for Java API compatibility.
  override def setMaxDepth(value: Int): this.type = super.setMaxDepth(value)

  override def setMaxBins(value: Int): this.type = super.setMaxBins(value)

  override def setMinInstancesPerNode(value: Int): this.type =
    super.setMinInstancesPerNode(value)

  override def setMinInfoGain(value: Double): this.type = super.setMinInfoGain(value)

  override def setMaxMemoryInMB(value: Int): this.type = super.setMaxMemoryInMB(value)

  override def setCacheNodeIds(value: Boolean): this.type = super.setCacheNodeIds(value)

  override def setCheckpointInterval(value: Int): this.type = super.setCheckpointInterval(value)

  override def setImpurity(value: String): this.type = super.setImpurity(value)

  override def setSeed(value: Long): this.type = super.setSeed(value)

  override def copy(extra: ParamMap): YggdrasilRegressor = defaultCopy(extra)

  def train(
             input: RDD[LabeledPoint],
             transposedDataset: RDD[(Int, Array[Double])],
             categoricalFeatures: Map[Int, Int]): DecisionTreeRegressionModel = {
    val strategy = getOldStrategy(categoricalFeatures)
    val model = Yggdrasil.train(input, strategy, Some(transposedDataset), parentUID = Some(uid))
    model.asInstanceOf[DecisionTreeRegressionModel]
  }

  override protected def train(dataset: DataFrame): DecisionTreeRegressionModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    val strategy = getOldStrategy(categoricalFeatures)
    val model = Yggdrasil.train(oldDataset, strategy, colStoreInput = None, parentUID = Some(uid))
    model.asInstanceOf[DecisionTreeRegressionModel]
  }

  /** Create a Strategy instance to use with the old API. */
  private[impl] def getOldStrategy(categoricalFeatures: Map[Int, Int]): OldStrategy = {
    super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity,
      subsamplingRate = 1.0)
  }
}

/**
  * DecisionTree which partitions data by feature.
  *
  * Algorithm:
  *  - Repartition data, grouping by feature.
  *  - Prep data (sort continuous features).
  *  - On each partition, initialize instance--node map with each instance at root node.
  *  - Iterate, training 1 new level of the tree at a time:
  *     - On each partition, for each feature on the partition, select the best split for each node.
  *     - Aggregate best split for each node.
  *     - Aggregate bit vector (1 bit/instance) indicating whether each instance splits
  *       left or right.
  *     - Broadcast bit vector.  On each partition, update instance--node map.
  *
  * TODO: Update to use a sparse column store.
  */
private[ml] object Yggdrasil extends Logging {

  private[impl] class YggdrasilMetadata(
                                         val numClasses: Int,
                                         val maxBins: Int,
                                         val minInfoGain: Double,
                                         val impurity: Impurity,
                                         val categoricalFeaturesInfo: Map[Int, Int]) extends Serializable {

    private val unorderedSplits = {
      /**
        * borrowed from [[DecisionTreeMetadata.buildMetadata]]
        */
      if (numClasses > 2) {
        // Multiclass classification
        val maxCategoriesForUnorderedFeature =
          ((math.log(maxBins / 2 + 1) / math.log(2.0)) + 1).floor.toInt
        categoricalFeaturesInfo.filter { case (featureIndex, numCategories) =>
          numCategories > 1 && numCategories <= maxCategoriesForUnorderedFeature
        }.map { case (featureIndex, numCategories) =>
          // Hack: If a categorical feature has only 1 category, we treat it as continuous.
          // TODO(SPARK-9957): Handle this properly by filtering out those features.
          // Decide if some categorical features should be treated as unordered features,
          //  which require 2 * ((1 << numCategories - 1) - 1) bins.
          // We do this check with log values to prevent overflows in case numCategories is large.
          // The next check is equivalent to: 2 * ((1 << numCategories - 1) - 1) <= maxBins
          featureIndex -> findSplits(featureIndex, numCategories)
        }
      } else {
        Map.empty[Int, Array[ygg.CategoricalSplit]]
      }
    }

    /**
      * Returns all possible subsets of features for categorical splits.
      * Borrowed from [[RandomForest.findSplits]]
      */
    private def findSplits(
                            featureIndex: Int,
                            featureArity: Int): Array[ygg.CategoricalSplit] = {
      // Unordered features
      // 2^(featureArity - 1) - 1 combinations
      val numSplits = (1 << (featureArity - 1)) - 1
      val splits = new Array[ygg.CategoricalSplit](numSplits)

      var splitIndex = 0
      while (splitIndex < numSplits) {
        val categories: List[Double] =
          RandomForest.extractMultiClassCategories(splitIndex + 1, featureArity)
        splits(splitIndex) =
          new ygg.CategoricalSplit(featureIndex, categories.toArray, featureArity)
        splitIndex += 1
      }
      splits
    }

    def getUnorderedSplits(featureIndex: Int): Array[ygg.CategoricalSplit] = unorderedSplits(featureIndex)

    def isClassification: Boolean = numClasses >= 2

    def isMulticlass: Boolean = numClasses > 2

    def isUnorderedFeature(featureIndex: Int): Boolean = unorderedSplits.contains(featureIndex)

    def createImpurityAggregator(): ImpurityAggregatorSingle = {
      impurity match {
        case Entropy => new EntropyAggregatorSingle(numClasses)
        case Gini => new GiniAggregatorSingle(numClasses)
        case Variance => new VarianceAggregatorSingle
      }
    }
  }

  private[impl] object YggdrasilMetadata {
    def fromStrategy(strategy: OldStrategy): YggdrasilMetadata = new YggdrasilMetadata(strategy.numClasses,
      strategy.maxBins, strategy.minInfoGain, strategy.impurity, strategy.categoricalFeaturesInfo)
  }

  /**
    * Method to train a decision tree model over an RDD.
    */
  def train(
             input: RDD[LabeledPoint],
             strategy: OldStrategy,
             colStoreInput: Option[RDD[(Int, Array[Double])]] = None,
             parentUID: Option[String] = None): DecisionTreeModel = {
    // TODO: Check validity of params
    // TODO: Check for empty dataset
    val numFeatures = input.first().features.size
    val rootNode = trainImpl(input, strategy, colStoreInput)
    finalizeTree(rootNode, strategy.algo, strategy.numClasses, numFeatures,
      parentUID)
  }

  private[impl] def finalizeTree(
                                  rootNode: SparkNode,
                                  algo: OldAlgo.Algo,
                                  numClasses: Int,
                                  numFeatures: Int,
                                  parentUID: Option[String]): DecisionTreeModel = {
    parentUID match {
      case Some(uid) =>
        if (algo == OldAlgo.Classification) {
          new DecisionTreeClassificationModel(uid, rootNode, numFeatures = numFeatures,
            numClasses = numClasses)
        } else {
          new DecisionTreeRegressionModel(uid, rootNode, numFeatures = numFeatures)
        }
      case None =>
        if (algo == OldAlgo.Classification) {
          new DecisionTreeClassificationModel(rootNode, numFeatures = numFeatures,
            numClasses = numClasses)
        } else {
          new DecisionTreeRegressionModel(rootNode, numFeatures = numFeatures)
        }
    }
  }

  private[impl] def getPredict(impurityCalculator: ImpurityCalculator): Predict = {
    val pred = impurityCalculator.predict
    new Predict(predict = pred, prob = impurityCalculator.prob(pred))
  }

  private[impl] def trainImpl(
                               input: RDD[LabeledPoint],
                               strategy: OldStrategy,
                               colStoreInput: Option[RDD[(Int, Array[Double])]]): SparkNode = {
    val metadata = YggdrasilMetadata.fromStrategy(strategy)

    // The case with 1 node (depth = 0) is handled separately.
    // This allows all iterations in the depth > 0 case to use the same code.
    // TODO: Check that learning works when maxDepth > 0 but learning stops at 1 node (because of
    //       other parameters).
    if (strategy.maxDepth == 0) {
      val impurityAggregator: ImpurityAggregatorSingle =
        input.aggregate(metadata.createImpurityAggregator())(
          (agg, lp) => agg.update(lp.label, 1.0),
          (agg1, agg2) => agg1.add(agg2))
      val impurityCalculator = impurityAggregator.getCalculator
      return new LeafNode(getPredict(impurityCalculator).predict, impurityCalculator.calculate(),
        impurityCalculator)
    }

    // Prepare column store.
    //   Note: rowToColumnStoreDense checks to make sure numRows < Int.MaxValue.
    // TODO: Is this mapping from arrays to iterators to arrays (when constructing learningData)?
    //       Or is the mapping implicit (i.e., not costly)?
    val colStoreInit: RDD[(Int, Array[Double])] = colStoreInput.getOrElse(
      rowToColumnStoreDense(input.map(_.features)))
    val numRows: Int = colStoreInit.first()._2.length
    if (metadata.numClasses > 1 && metadata.numClasses <= 32) {
      YggdrasilClassification.trainImpl(input, colStoreInit, metadata, numRows, strategy.maxDepth)
    } else {
      YggdrasilRegression.trainImpl(input, colStoreInit, metadata, numRows, strategy.maxDepth)
    }
  }


  /**
    * On driver: Grow tree based on chosen splits, and compute new set of active nodes.
    * @param oldPeriphery  Old periphery of active nodes.
    * @param bestSplitsAndGains  Best (split, gain) pairs, which can be zipped with the old
    *                            periphery.  These stats will be used to replace the stats in
    *                            any nodes which are split.
    * @param minInfoGain  Threshold for min info gain required to split a node.
    * @return  New active node periphery.
    *          If a node is split, then this method will update its fields.
    */
  private[impl] def computeActiveNodePeriphery(
                                                oldPeriphery: Array[ygg.LearningNode],
                                                bestSplitsAndGains: Array[(Option[ygg.Split], ImpurityStats)],
                                                minInfoGain: Double): Array[ygg.LearningNode] = {
    bestSplitsAndGains.zipWithIndex.flatMap {
      case ((split, stats), nodeIdx) =>
        val node = oldPeriphery(nodeIdx)
        if (split.nonEmpty && stats.gain > minInfoGain) {
          // TODO: remove node id
          node.leftChild = Some(ygg.LearningNode(node.id * 2, isLeaf = false,
            new ImpurityStats(Double.NaN, stats.leftImpurity, stats.leftImpurityCalculator,
              null, null, true)))
          node.rightChild = Some(ygg.LearningNode(node.id * 2 + 1, isLeaf = false,
            new ImpurityStats(Double.NaN, stats.rightImpurity, stats.rightImpurityCalculator,
              null, null, true)))
          node.split = split
          node.isLeaf = false
          node.stats = stats
          Iterator(node.leftChild.get, node.rightChild.get)
        } else {
          node.isLeaf = true
          Iterator()
        }
    }
  }

  /**
    * Aggregate bit vector (1 bit/instance) indicating whether each instance goes left/right.
    * - Send chosen splits to workers.
    * - Each worker creates part of the bit vector corresponding to the splits it created.
    * - Aggregate the partial bit vectors to create one vector (of length numRows).
    *   Correction: Aggregate only the pieces of that vector corresponding to instances at
    *   active nodes.
    * @param partitionInfos  RDD with feature data, plus current status metadata
    * @param bestSplits  Split for each active node, or None if that node will not be split
    * @return Array of bit vectors, ordered by offset ranges
    */
  private[impl] def aggregateBitVector(
                                        partitionInfos: RDD[PartitionInfo],
                                        bestSplits: Array[Option[ygg.Split]],
                                        numRows: Int): RoaringBitmap = {
    val bestSplitsBc: Broadcast[Array[Option[ygg.Split]]] =
      partitionInfos.sparkContext.broadcast(bestSplits)
    val workerBitSubvectors: RDD[RoaringBitmap] = partitionInfos.map {
      case PartitionInfo(columns: Array[FeatureVector], nodeOffsets: Array[Int],
      activeNodes: BitSet, fullImpurities: Array[ImpurityAggregatorSingle]) =>
        val localBestSplits: Array[Option[ygg.Split]] = bestSplitsBc.value
        // localFeatureIndex[feature index] = index into PartitionInfo.columns
        val localFeatureIndex: Map[Int, Int] = columns.map(_.featureIndex).zipWithIndex.toMap
        val bitSetForNodes: Iterator[RoaringBitmap] = activeNodes.iterator
          .zip(localBestSplits.iterator).flatMap {
          case (nodeIndexInLevel: Int, Some(split: ygg.Split)) =>
            if (localFeatureIndex.contains(split.featureIndex)) {
              // This partition has the column (feature) used for this split.
              val fromOffset = nodeOffsets(nodeIndexInLevel)
              val toOffset = nodeOffsets(nodeIndexInLevel + 1)
              val colIndex: Int = localFeatureIndex(split.featureIndex)
              Iterator(bitVectorFromSplit(columns(colIndex), fromOffset, toOffset, split, numRows))
            } else {
              Iterator()
            }
          case (nodeIndexInLevel: Int, None) =>
            // Do not create a bitVector when there is no split.
            // PartitionInfo.update will detect that there is no
            // split, by how many instances go left/right.
            Iterator()
        }
        if (bitSetForNodes.isEmpty) {
          new RoaringBitmap()
        } else {
          bitSetForNodes.reduce[RoaringBitmap] { (acc, bitv) => acc.or(bitv); acc }
        }
    }
    val aggBitVector: RoaringBitmap = workerBitSubvectors.reduce { (acc, bitv) =>
      acc.or(bitv)
      acc
    }
    bestSplitsBc.unpersist()
    aggBitVector
  }

  /**
    * For a given feature, for a given node, apply a split and return a bit vector indicating the
    * outcome of the split for each instance at that node.
    *
    * @param col  Column for feature
    * @param from  Start offset in col for the node
    * @param to  End offset in col for the node
    * @param split  Split to apply to instances at this node.
    * @return  Bits indicating splits for instances at this node.
    *          These bits are sorted by the row indices, in order to guarantee an ordering
    *          understood by all workers.
    *          Thus, the bit indices used are based on 2-level sorting: first by node, and
    *          second by sorted row indices within the node's rows.
    *          bit[index in sorted array of row indices] = false for left, true for right
    */
  private[impl] def bitVectorFromSplit(
                                        col: FeatureVector,
                                        from: Int,
                                        to: Int,
                                        split: ygg.Split,
                                        numRows: Int): RoaringBitmap = {
    val bitv = new RoaringBitmap()
    var i = from
    while (i < to) {
      val value = col.values(i)
      val idx = col.indices(i)
      if (!split.shouldGoLeft(value)) {
        bitv.add(idx)
      }
      i += 1
    }
    bitv
  }

  /**
    * Intermediate data stored on each partition during learning.
    *
    * Node indexing for nodeOffsets, activeNodes:
    * Nodes are indexed left-to-right along the periphery of the tree, with 0-based indices.
    * The periphery is the set of leaf nodes (active and inactive).
    *
    * @param columns  Subset of columns (features) stored in this partition.
    *                 Each column is sorted first by nodes (left-to-right along the tree periphery);
    *                 all columns share this first level of sorting.
    *                 Within each node's group, each column is sorted based on feature value;
    *                 this second level of sorting differs across columns.
    * @param nodeOffsets  Offsets into the columns indicating the first level of sorting (by node).
    *                     The rows corresponding to node i are in the range
    *                     [nodeOffsets(i), nodeOffsets(i+1)).
    * @param activeNodes  Nodes which are active (still being split).
    *                     Inactive nodes are known to be leafs in the final tree.
    *                     TODO: Should this (and even nodeOffsets) not be stored in PartitionInfo,
    *                           but instead on the driver?
    */
  private[impl] case class PartitionInfo(
                                          columns: Array[FeatureVector],
                                          nodeOffsets: Array[Int],
                                          activeNodes: BitSet,
                                          fullImpurityAggs: Array[ImpurityAggregatorSingle]) extends Serializable {

    // pre-allocated temporary buffers that we use to sort
    // instances in left and right children during update
    val tempVals: Array[Double] = new Array[Double](columns(0).values.length)
    val tempIndices: Array[Int] = new Array[Int](columns(0).values.length)

    /** For debugging */
    override def toString: String = {
      "PartitionInfo(" +
        "  columns: {\n" +
        columns.mkString(",\n") +
        "  },\n" +
        s"  nodeOffsets: ${nodeOffsets.mkString(", ")},\n" +
        s"  activeNodes: ${activeNodes.iterator.mkString(", ")},\n" +
        ")\n"
    }

    /**
      * Update columns and nodeOffsets for the next level of the tree.
      *
      * Update columns:
      *   For each column,
      *     For each (previously) active node,
      *       Sort corresponding range of instances based on bit vector.
      * Update nodeOffsets, activeNodes:
      *   Split offsets for nodes which split (which can be identified using the bit vector).
      *
      * @param instanceBitVector  Bit vector encoding splits for the next level of the tree.
      *                    These must follow a 2-level ordering, where the first level is by node
      *                    and the second level is by row index.
      *                    bitVector(i) = false iff instance i goes to the left child.
      *                    For instances at inactive (leaf) nodes, the value can be arbitrary.
      * @return Updated partition info
      */
    def update(instanceBitVector: BitSet, newNumNodeOffsets: Int,
               labels: Array[Byte], metadata: YggdrasilMetadata): PartitionInfo = {
      // Create a 2-level representation of the new nodeOffsets (to be flattened).
      // These 2 levels correspond to original nodes and their children (if split).
      val newNodeOffsets = nodeOffsets.map(Array(_))
      val newFullImpurityAggs = fullImpurityAggs.map(Array(_))

      val newColumns = columns.zipWithIndex.map { case (col, index) =>
        index match {
          case 0 => first(col, instanceBitVector, metadata, labels, newNodeOffsets, newFullImpurityAggs)
          case _ => rest(col, instanceBitVector, newNodeOffsets)
        }
        col
      }

      // Identify the new activeNodes based on the 2-level representation of the new nodeOffsets.
      val newActiveNodes = new BitSet(newNumNodeOffsets - 1)
      var newNodeOffsetsIdx = 0
      var i = 0
      while (i < newNodeOffsets.length) {
        val offsets = newNodeOffsets(i)
        if (offsets.length == 2) {
          newActiveNodes.set(newNodeOffsetsIdx)
          newActiveNodes.set(newNodeOffsetsIdx + 1)
          newNodeOffsetsIdx += 2
        } else {
          newNodeOffsetsIdx += 1
        }
        i += 1
      }
      PartitionInfo(newColumns, newNodeOffsets.flatten, newActiveNodes, newFullImpurityAggs.flatten)
    }


    /**
      * Sort the very first column in the [[PartitionInfo.columns]]. While
      * we sort the column, we also update [[PartitionInfo.nodeOffsets]]
      * (by modifying @param newNodeOffsets) and [[PartitionInfo.fullImpurityAggs]]
      * (by modifying @param newFullImpurityAggs).
      * @param col The very first column in [[PartitionInfo.columns]]
      * @param metadata Used to create new [[ImpurityAggregatorSingle]] for a new child
      *                 node in the tree
      * @param labels   Labels are read as we sort column to populate stats for each
      *                 new ImpurityAggregatorSingle
      */
    private def first(
                       col: FeatureVector,
                       instanceBitVector: BitSet,
                       metadata: YggdrasilMetadata,
                       labels: Array[Byte],
                       newNodeOffsets: Array[Array[Int]],
                       newFullImpurityAggs: Array[Array[ImpurityAggregatorSingle]]) = {
      activeNodes.iterator.foreach { nodeIdx =>
        // WHAT TO OPTIMIZE:
        // - try skipping numBitsSet
        // - maybe uncompress bitmap
        val from = nodeOffsets(nodeIdx)
        val to = nodeOffsets(nodeIdx + 1)

        // If this is the very first time we split,
        // we don't use rangeIndices to count the number of bits set;
        // the entire bit vector will be used, so getCardinality
        // will give us the same result more cheaply.
        val numBitsSet = {
          if (nodeOffsets.length == 2) instanceBitVector.cardinality()
          else {
            var count = 0
            var i = from
            while (i < to) {
              if (instanceBitVector.get(col.indices(i))) {
                count += 1
              }
              i += 1
            }
            count
          }
        }

        val numBitsNotSet = to - from - numBitsSet // number of instances splitting left
      val oldOffset = newNodeOffsets(nodeIdx).head

        // If numBitsNotSet or numBitsSet equals 0, then this node was not split,
        // so we do not need to update its part of the column. Otherwise, we update it.
        if (numBitsNotSet != 0 && numBitsSet != 0) {
          newNodeOffsets(nodeIdx) = Array(oldOffset, oldOffset + numBitsNotSet)

          val leftImpurity = metadata.createImpurityAggregator()
          val rightImpurity = metadata.createImpurityAggregator()

          // BEGIN SORTING
          // We sort the [from, to) slice of col based on instance bit, then
          // instance value. This is required to match the bit vector across all
          // workers. All instances going "left" in the split (which are false)
          // should be ordered before the instances going "right". The instanceBitVector
          // gives us the bit value for each instance based on the instance's index.
          // Then both [from, numBitsNotSet) and [numBitsNotSet, to) need to be sorted
          // by value.
          // Since the column is already sorted by value, we can compute
          // this sort in a single pass over the data. We iterate from start to finish
          // (which preserves the sorted order), and then copy the values
          // into @tempVals and @tempIndices either:
          // 1) in the [from, numBitsNotSet) range if the bit is false, or
          // 2) in the [numBitsNotSet, to) range if the bit is true.
          var (leftInstanceIdx, rightInstanceIdx) = (from, from + numBitsNotSet)
          var idx = from
          while (idx < to) {
            val indexForVal = col.indices(idx)
            val bit = instanceBitVector.get(indexForVal)
            val label = labels(indexForVal)
            if (bit) {
              rightImpurity.update(label)
              tempVals(rightInstanceIdx) = col.values(idx)
              tempIndices(rightInstanceIdx) = indexForVal
              rightInstanceIdx += 1
            } else {
              leftImpurity.update(label)
              tempVals(leftInstanceIdx) = col.values(idx)
              tempIndices(leftInstanceIdx) = indexForVal
              leftInstanceIdx += 1
            }
            idx += 1
          }
          // END SORTING

          newFullImpurityAggs(nodeIdx) = Array(leftImpurity, rightImpurity)
          // update the column values and indices
          // with the corresponding indices
          System.arraycopy(tempVals, from, col.values, from, to - from)
          System.arraycopy(tempIndices, from, col.indices, from, to - from)
        }
      }
    }

    /**
      * Update columns and nodeOffsets for the next level of the tree.
      *
      * Update columns:
      *   For each column,
      *     For each (previously) active node,
      *       Sort corresponding range of instances based on bit vector.
      * Update nodeOffsets, activeNodes:
      *   Split offsets for nodes which split (which can be identified using the bit vector).
      *
      * @param instanceBitVector  Bit vector encoding splits for the next level of the tree.
      *                    These must follow a 2-level ordering, where the first level is by node
      *                    and the second level is by row index.
      *                    bitVector(i) = false iff instance i goes to the left child.
      *                    For instances at inactive (leaf) nodes, the value can be arbitrary.
      * @return Updated partition info
      */
    def update(instanceBitVector: BitSet, newNumNodeOffsets: Int,
               labels: Array[Double], metadata: YggdrasilMetadata): PartitionInfo = {
      // Create a 2-level representation of the new nodeOffsets (to be flattened).
      // These 2 levels correspond to original nodes and their children (if split).
      val newNodeOffsets = nodeOffsets.map(Array(_))
      val newFullImpurityAggs = fullImpurityAggs.map(Array(_))

      val newColumns = columns.zipWithIndex.map { case (col, index) =>
        index match {
          case 0 => first(col, instanceBitVector, metadata, labels, newNodeOffsets, newFullImpurityAggs)
          case _ => rest(col, instanceBitVector, newNodeOffsets)
        }
        col
      }

      // Identify the new activeNodes based on the 2-level representation of the new nodeOffsets.
      val newActiveNodes = new BitSet(newNumNodeOffsets - 1)
      var newNodeOffsetsIdx = 0
      var i = 0
      while (i < newNodeOffsets.length) {
        val offsets = newNodeOffsets(i)
        if (offsets.length == 2) {
          newActiveNodes.set(newNodeOffsetsIdx)
          newActiveNodes.set(newNodeOffsetsIdx + 1)
          newNodeOffsetsIdx += 2
        } else {
          newNodeOffsetsIdx += 1
        }
        i += 1
      }
      PartitionInfo(newColumns, newNodeOffsets.flatten, newActiveNodes, newFullImpurityAggs.flatten)
    }


    /**
      * Sort the very first column in the [[PartitionInfo.columns]]. While
      * we sort the column, we also update [[PartitionInfo.nodeOffsets]]
      * (by modifying @param newNodeOffsets) and [[PartitionInfo.fullImpurityAggs]]
      * (by modifying @param newFullImpurityAggs).
      * @param col The very first column in [[PartitionInfo.columns]]
      * @param metadata Used to create new [[ImpurityAggregatorSingle]] for a new child
      *                 node in the tree
      * @param labels   Labels are read as we sort column to populate stats for each
      *                 new ImpurityAggregatorSingle
      */
    private def first(
                       col: FeatureVector,
                       instanceBitVector: BitSet,
                       metadata: YggdrasilMetadata,
                       labels: Array[Double],
                       newNodeOffsets: Array[Array[Int]],
                       newFullImpurityAggs: Array[Array[ImpurityAggregatorSingle]]) = {
      activeNodes.iterator.foreach { nodeIdx =>
        // WHAT TO OPTIMIZE:
        // - try skipping numBitsSet
        // - maybe uncompress bitmap
        val from = nodeOffsets(nodeIdx)
        val to = nodeOffsets(nodeIdx + 1)

        // If this is the very first time we split,
        // we don't use rangeIndices to count the number of bits set;
        // the entire bit vector will be used, so getCardinality
        // will give us the same result more cheaply.
        val numBitsSet = {
          if (nodeOffsets.length == 2) instanceBitVector.cardinality()
          else {
            var count = 0
            var i = from
            while (i < to) {
              if (instanceBitVector.get(col.indices(i))) {
                count += 1
              }
              i += 1
            }
            count
          }
        }

        val numBitsNotSet = to - from - numBitsSet // number of instances splitting left
      val oldOffset = newNodeOffsets(nodeIdx).head

        // If numBitsNotSet or numBitsSet equals 0, then this node was not split,
        // so we do not need to update its part of the column. Otherwise, we update it.
        if (numBitsNotSet != 0 && numBitsSet != 0) {
          newNodeOffsets(nodeIdx) = Array(oldOffset, oldOffset + numBitsNotSet)

          val leftImpurity = metadata.createImpurityAggregator()
          val rightImpurity = metadata.createImpurityAggregator()

          // BEGIN SORTING
          // We sort the [from, to) slice of col based on instance bit, then
          // instance value. This is required to match the bit vector across all
          // workers. All instances going "left" in the split (which are false)
          // should be ordered before the instances going "right". The instanceBitVector
          // gives us the bit value for each instance based on the instance's index.
          // Then both [from, numBitsNotSet) and [numBitsNotSet, to) need to be sorted
          // by value.
          // Since the column is already sorted by value, we can compute
          // this sort in a single pass over the data. We iterate from start to finish
          // (which preserves the sorted order), and then copy the values
          // into @tempVals and @tempIndices either:
          // 1) in the [from, numBitsNotSet) range if the bit is false, or
          // 2) in the [numBitsNotSet, to) range if the bit is true.
          var (leftInstanceIdx, rightInstanceIdx) = (from, from + numBitsNotSet)
          var idx = from
          while (idx < to) {
            val indexForVal = col.indices(idx)
            val bit = instanceBitVector.get(indexForVal)
            val label = labels(indexForVal)
            if (bit) {
              rightImpurity.update(label)
              tempVals(rightInstanceIdx) = col.values(idx)
              tempIndices(rightInstanceIdx) = indexForVal
              rightInstanceIdx += 1
            } else {
              leftImpurity.update(label)
              tempVals(leftInstanceIdx) = col.values(idx)
              tempIndices(leftInstanceIdx) = indexForVal
              leftInstanceIdx += 1
            }
            idx += 1
          }
          // END SORTING

          newFullImpurityAggs(nodeIdx) = Array(leftImpurity, rightImpurity)
          // update the column values and indices
          // with the corresponding indices
          System.arraycopy(tempVals, from, col.values, from, to - from)
          System.arraycopy(tempIndices, from, col.indices, from, to - from)
        }
      }
    }

    /**
      * Sort the remaining columns in the [[PartitionInfo.columns]]. Since
      * we already computed [[PartitionInfo.nodeOffsets]] and
      * [[PartitionInfo.fullImpurityAggs]] while we sorted the first column,
      * we skip the computation for those here.
      * @param col The very first column in [[PartitionInfo.columns]]
      * @param newNodeOffsets Instead of re-computing number of bits set/not set
      *                       per split, we read those values from here
      */
    private def rest(
                      col: FeatureVector,
                      instanceBitVector: BitSet,
                      newNodeOffsets: Array[Array[Int]]) = {
      activeNodes.iterator.foreach { nodeIdx =>
        val from = nodeOffsets(nodeIdx)
        val to = nodeOffsets(nodeIdx + 1)
        val newOffsets = newNodeOffsets(nodeIdx)

        // We determined that this node was split in first()
        if (newOffsets.length == 2) {
          val numBitsNotSet = newOffsets(1) - newOffsets(0)

          // Same as above, but we don't compute the left and right impurities for
          // the resulitng child nodes
          var (leftInstanceIdx, rightInstanceIdx) = (from, from + numBitsNotSet)
          var idx = from
          while (idx < to) {
            val indexForVal = col.indices(idx)
            val bit = instanceBitVector.get(indexForVal)
            if (bit) {
              tempVals(rightInstanceIdx) = col.values(idx)
              tempIndices(rightInstanceIdx) = indexForVal
              rightInstanceIdx += 1
            } else {
              tempVals(leftInstanceIdx) = col.values(idx)
              tempIndices(leftInstanceIdx) = indexForVal
              leftInstanceIdx += 1
            }
            idx += 1
          }

          System.arraycopy(tempVals, from, col.values, from, to - from)
          System.arraycopy(tempIndices, from, col.indices, from, to - from)
        }
      }
    }

  }

  /**
    * Feature vector types are based on (feature type, representation).
    * The feature type can be continuous or categorical.
    *
    * Features are sorted by value, so we must store indices + values.
    * These values are currently stored in a dense representation only.
    * TODO: Support sparse storage (to optimize deeper levels of the tree), and maybe compressed
    *       storage (to optimize upper levels of the tree).
    * @param featureArity  For categorical features, this gives the number of categories.
    *                      For continuous features, this should be set to 0.
    */
  private[impl] class FeatureVector(
                                     val featureIndex: Int,
                                     val featureArity: Int,
                                     val values: Array[Double],
                                     val indices: Array[Int])
    extends Serializable {

    def isCategorical: Boolean = featureArity > 0

    /** For debugging */
    override def toString: String = {
      "  FeatureVector(" +
        s"    featureIndex: $featureIndex,\n" +
        s"    featureType: ${if (featureArity == 0) "Continuous" else "Categorical"},\n" +
        s"    featureArity: $featureArity,\n" +
        s"    values: ${values.mkString(", ")},\n" +
        s"    indices: ${indices.mkString(", ")},\n" +
        "  )"
    }

    def deepCopy(): FeatureVector =
      new FeatureVector(featureIndex, featureArity, values.clone(), indices.clone())

    override def equals(other: Any): Boolean = {
      other match {
        case o: FeatureVector =>
          featureIndex == o.featureIndex && featureArity == o.featureArity &&
            values.sameElements(o.values) && indices.sameElements(o.indices)
        case _ => false
      }
    }
  }

  private[impl] object FeatureVector {
    /** Store column sorted by feature values. */
    def fromOriginal(
                      featureIndex: Int,
                      featureArity: Int,
                      values: Array[Double]): FeatureVector = {
      val indices = values.indices.toArray
      val fv = new FeatureVector(featureIndex, featureArity, values, indices)
      val sorter = new Sorter(new FeatureVectorSortByValue(featureIndex, featureArity))
      sorter.sort(fv, 0, values.length, Ordering[KeyWrapper])
      fv
    }
  }

  /**
    * Sort FeatureVector by values column; @see [[FeatureVector.fromOriginal()]]
    * @param featureIndex @param featureArity Passed in so that, if a new
    *                     FeatureVector is allocated during sorting, that new object
    *                     also has the same featureIndex and featureArity
    */
  private class FeatureVectorSortByValue(featureIndex: Int, featureArity: Int)
    extends SortDataFormat[KeyWrapper, FeatureVector] {

    override def newKey(): KeyWrapper = new KeyWrapper()

    override def getKey(data: FeatureVector,
                        pos: Int,
                        reuse: KeyWrapper): KeyWrapper = {
      if (reuse == null) {
        new KeyWrapper().setKey(data.values(pos))
      } else {
        reuse.setKey(data.values(pos))
      }
    }

    override def getKey(data: FeatureVector,
                        pos: Int): KeyWrapper = {
      getKey(data, pos, null)
    }

    private def swapElements(data: Array[Double],
                             pos0: Int,
                             pos1: Int): Unit = {
      val tmp = data(pos0)
      data(pos0) = data(pos1)
      data(pos1) = tmp
    }

    private def swapElements(data: Array[Int],
                             pos0: Int,
                             pos1: Int): Unit = {
      val tmp = data(pos0)
      data(pos0) = data(pos1)
      data(pos1) = tmp
    }

    override def swap(data: FeatureVector, pos0: Int, pos1: Int): Unit = {
      swapElements(data.values, pos0, pos1)
      swapElements(data.indices, pos0, pos1)
    }

    override def copyRange(src: FeatureVector,
                           srcPos: Int,
                           dst: FeatureVector,
                           dstPos: Int,
                           length: Int): Unit = {
      System.arraycopy(src.values, srcPos, dst.values, dstPos, length)
      System.arraycopy(src.indices, srcPos, dst.indices, dstPos, length)
    }

    override def allocate(length: Int): FeatureVector = {
      new FeatureVector(featureIndex, featureArity, new Array[Double](length), new Array[Int](length))
    }

    override def copyElement(src: FeatureVector,
                             srcPos: Int,
                             dst: FeatureVector,
                             dstPos: Int): Unit = {
      dst.values(dstPos) = src.values(srcPos)
      dst.indices(dstPos) = src.indices(srcPos)
    }
  }

  /**
    * A wrapper that holds a primitive key – borrowed from [[org.apache.spark.ml.recommendation.ALS]]
    */
  private class KeyWrapper extends Ordered[KeyWrapper] {

    var key: Double = _

    override def compare(that: KeyWrapper): Int = {
      scala.math.Ordering.Double.compare(key, that.key)
    }

    def setKey(key: Double): this.type = {
      this.key = key
      this
    }
  }
}
