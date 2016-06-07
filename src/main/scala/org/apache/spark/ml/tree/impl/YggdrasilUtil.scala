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

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD


@DeveloperApi
object YggdrasilUtil {

  /**
   * Convert a dataset of [[Vector]] from row storage to column storage.
   * This can take any [[Vector]] type but stores data as [[DenseVector]].
   *
   * WARNING: This shuffles the ENTIRE dataset across the network, so it is a VERY EXPENSIVE
   *          operation.  This can also fail if 1 column is too large to fit on 1 partition.
   *
   * This maintains sparsity in the data.
   *
   * This maintains matrix structure.  I.e., each partition of the output RDD holds adjacent
   * columns.  The number of partitions will be min(input RDD's number of partitions, numColumns).
   *
   * @param rowStore  The input vectors are data rows/instances.
   * @return RDD of (columnIndex, columnValues) pairs,
   *         where each pair corresponds to one entire column.
   *         If either dimension of the given data is 0, this returns an empty RDD.
   *         If vector lengths do not match, this throws an exception.
   *
   * TODO: Add implementation for sparse data.
   *       For sparse data, distribute more evenly based on number of non-zeros.
   *       (First collect stats to decide how to partition.)
   * TODO: Move elsewhere in MLlib.
   */
  def rowToColumnStoreDense(rowStore: RDD[Vector]): RDD[(Int, Array[Double])] = {

    val numRows = {
      val longNumRows: Long = rowStore.count()
      require(longNumRows < Int.MaxValue, s"rowToColumnStore given RDD with $longNumRows rows," +
        s" but can handle at most ${Int.MaxValue} rows")
      longNumRows.toInt
    }
    if (numRows == 0) {
      return rowStore.sparkContext.parallelize(Seq.empty[(Int, Array[Double])])
    }
    val numCols = rowStore.take(1)(0).size
    if (numCols == 0) {
      return rowStore.sparkContext.parallelize(Seq.empty[(Int, Array[Double])])
    }

    val numSourcePartitions = rowStore.partitions.length
    val approxNumTargetPartitions = Math.min(numCols, numSourcePartitions)
    val maxColumnsPerPartition = Math.ceil(numCols / approxNumTargetPartitions.toDouble).toInt
    val numTargetPartitions = Math.ceil(numCols / maxColumnsPerPartition.toDouble).toInt

    def getNumColsInGroup(groupIndex: Int) = {
      if (groupIndex + 1 < numTargetPartitions) {
        maxColumnsPerPartition
      } else {
        numCols - (numTargetPartitions - 1) * maxColumnsPerPartition // last partition
      }
    }

    /* On each partition, re-organize into groups of columns:
         (groupIndex, (sourcePartitionIndex, partCols)),
         where partCols(colIdx) = partial column.
       The groupIndex will be used to groupByKey.
       The sourcePartitionIndex is used to ensure instance indices match up after the shuffle.
       The partial columns will be stacked into full columns after the shuffle.
       Note: By design, partCols will always have at least 1 column.
     */
    val partialColumns: RDD[(Int, (Int, Array[Array[Double]]))] =
      rowStore.mapPartitionsWithIndex { case (sourcePartitionIndex, iterator) =>
        // columnSets(groupIndex)(colIdx)
        //   = column values for each instance in sourcePartitionIndex,
        // where colIdx is a 0-based index for columns for groupIndex
        val columnSets = new Array[Array[ArrayBuffer[Double]]](numTargetPartitions)
        var groupIndex = 0
        while(groupIndex < numTargetPartitions) {
          columnSets(groupIndex) =
            Array.fill[ArrayBuffer[Double]](getNumColsInGroup(groupIndex))(ArrayBuffer[Double]())
          groupIndex += 1
        }
        while (iterator.hasNext) {
          val row = iterator.next.toArray
          var groupIndex = 0
          while (groupIndex < numTargetPartitions) {
            val fromCol = groupIndex * maxColumnsPerPartition
            val numColsInTargetPartition = getNumColsInGroup(groupIndex)
            // TODO: match-case here on row as Dense or Sparse Vector (for speed)
            var colIdx = 0
            while (colIdx < numColsInTargetPartition) {
              columnSets(groupIndex)(colIdx) += row(fromCol + colIdx)
              colIdx += 1
            }
            groupIndex += 1
          }
        }
        Range(0, numTargetPartitions).map { groupIndex =>
          (groupIndex, (sourcePartitionIndex, columnSets(groupIndex).map(_.toArray)))
        }.toIterator
      }

    // Shuffle data
    val groupedPartialColumns: RDD[(Int, Iterable[(Int, Array[Array[Double]])])] =
      partialColumns.groupByKey()

    // Each target partition now holds its set of columns.
    // Group the partial columns into full columns.
    val fullColumns = groupedPartialColumns.flatMap { case (groupIndex, iterable) =>
      // We do not know the number of rows per group, so we need to collect the groups
      // before filling the full columns.
      val collectedPartCols = new Array[Array[Array[Double]]](numSourcePartitions)
      val iter = iterable.iterator
      while (iter.hasNext) {
        val (sourcePartitionIndex, partCols) = iter.next()
        collectedPartCols(sourcePartitionIndex) = partCols
      }
      val rowOffsets: Array[Int] = collectedPartCols.map(_(0).length).scanLeft(0)(_ + _)
      val numRows = rowOffsets.last
      // Initialize full columns
      val fromCol = groupIndex * maxColumnsPerPartition
      val numColumnsInPartition = getNumColsInGroup(groupIndex)
      val partitionColumns: Array[Array[Double]] =
        Array.fill[Array[Double]](numColumnsInPartition)(new Array[Double](numRows))
      var colIdx = 0 // index within group
      while (colIdx < numColumnsInPartition) {
        var sourcePartitionIndex = 0
        while (sourcePartitionIndex < numSourcePartitions) {
          val partColLength =
            rowOffsets(sourcePartitionIndex + 1) - rowOffsets(sourcePartitionIndex)
          Array.copy(collectedPartCols(sourcePartitionIndex)(colIdx), 0,
            partitionColumns(colIdx), rowOffsets(sourcePartitionIndex), partColLength)
          sourcePartitionIndex += 1
        }
        colIdx += 1
      }
      val columnIndices = Range(0, numColumnsInPartition).map(_ + fromCol)
//      val columns = partitionColumns.map(Vectors.dense)
      columnIndices.zip(partitionColumns)
    }

    fullColumns
  }

  /**
   * This checks for an empty RDD (0 rows or 0 columns).
   * This will throw an exception if any columns have non-matching numbers of features.
   * @param rowStore  Dataset of vectors which all have the same length (number of columns).
   * @return  Array over columns of the number of non-zero elements in each column.
   *          Returns empty array if the RDD is empty.
   */
  private def countNonZerosPerColumn(rowStore: RDD[Vector]): Array[Long] = {
    val firstRow = rowStore.take(1)
    if (firstRow.length == 0) {
      return Array.empty[Long]
    }
    val numCols = firstRow(0).size
    val colSizes: Array[Long] = rowStore.mapPartitions { iterator =>
      val partColSizes = Array.fill[Long](numCols)(0)
      iterator.foreach {
        case dv: DenseVector =>
          var col = 0
          while (col < dv.size) {
            if (dv(col) != 0.0) partColSizes(col) += 1
            col += 1
          }
        case sv: SparseVector =>
          var k = 0
          while (k < sv.indices.length) {
            if (sv.values(k) != 0.0) partColSizes(sv.indices(k)) += 1
            k += 1
          }
      }
      Iterator(partColSizes)
    }.fold(Array.fill[Long](numCols)(0)){
      case (v1, v2) => v1.zip(v2).map(v12 => v12._1 + v12._2)
    }
    colSizes
  }

  /**
   * The returned RDD sets the number of partitions as follows:
   *  - The targeted number is:
   *     numTargetPartitions = min(rowStore num partitions, num columns) * overPartitionFactor.
   *  - The actual number will be in the range [numTargetPartitions, 2 * numTargetPartitions].
   * Partitioning is done such that each partition holds consecutive columns.
   *
   * TODO: Update this to adaptively make columns dense or sparse based on a sparsity threshold.
   *
   * TODO: Cache rowStore temporarily.
   *
   * @param rowStore  RDD of dataset rows
   * @param overPartitionFactor  Multiplier for the targeted number of partitions.  This parameter
   *                             helps to ensure that P partitions handled by P compute cores
   *                             do not get split into slightly more than P partitions;
   *                             if that occurred, then work would not be shared evenly.
   * @return RDD of (column index, column) pairs
   */
  def rowToColumnStoreSparse(
      rowStore: RDD[Vector],
      overPartitionFactor: Int = 3): RDD[(Int, Vector)] = {

    val numRows = {
      val longNumRows: Long = rowStore.count()
      require(longNumRows < Int.MaxValue, s"rowToColumnStore given RDD with $longNumRows rows," +
        s" but can handle at most ${Int.MaxValue} rows")
      longNumRows.toInt
    }
    if (numRows == 0) {
      return rowStore.sparkContext.parallelize(Seq.empty[(Int, Vector)])
    }

    // Compute the number of non-zeros in each column.
    val colSizes: Array[Long] = countNonZerosPerColumn(rowStore)
    val numCols = colSizes.length
    val numSourcePartitions = rowStore.partitions.length
    if (numCols == 0 || numSourcePartitions == 0) {
      return rowStore.sparkContext.parallelize(Seq.empty[(Int, Vector)])
    }
    val totalNonZeros = colSizes.sum

    // Split columns into groups.
    // Groups are chosen greedily and sequentially, putting as many columns as possible in each
    //   group (limited by the number of non-zeros).  Try to limit the number of non-zeros per
    //   group to at most targetNonZerosPerPartition.
    val numTargetPartitions = math.min(numSourcePartitions, numCols) * overPartitionFactor
    val targetNonZerosPerPartition = (totalNonZeros / numTargetPartitions.toDouble).floor.toLong
    val groupStartColumns: Array[Int] = {
      val startCols = new ArrayBuffer[Int]()
      startCols += 0
      var currentStartCol = 0
      var currentNonZeros: Long = 0
      var col = 0
      while (col < numCols) {
        if (currentNonZeros >= targetNonZerosPerPartition && col != startCols.last) {
          startCols += col
          currentStartCol = col
          currentNonZeros = 0
        } else {
          currentNonZeros += colSizes(col)
        }
        col += 1
      }
      startCols += numCols
      startCols.toArray
    }
    val numGroups = groupStartColumns.length - 1 // actual number of destination partitions

    /* On each partition, re-organize into groups of columns:
         (groupIndex, (sourcePartitionIndex, partCols)),
         where partCols(colIdx) = partial column.
       The groupIndex will be used to groupByKey.
       The sourcePartitionIndex is used to ensure instance indices match up after the shuffle.
       The partial columns will be stacked into full columns after the shuffle.
       Note: By design, partCols will always have at least 1 column.
     */
    val partialColumns: RDD[(Int, (Int, Array[SparseVector]))] =
      rowStore.zipWithIndex().mapPartitionsWithIndex { case (sourcePartitionIndex, iterator) =>
        type SparseVectorBuffer = (Int, ArrayBuffer[Int], ArrayBuffer[Double])
        // columnSets(groupIndex)(colIdx)
        //   = column values for each instance in sourcePartitionIndex,
        // where colIdx is a 0-based index for columns for groupIndex,
        // and where column values are in sparse format: (size, indices, values)
        val columnSetSizes = new Array[Array[Int]](numGroups)
        val columnSetIndices = new Array[Array[ArrayBuffer[Int]]](numGroups)
        val columnSetValues = new Array[Array[ArrayBuffer[Double]]](numGroups)
        var groupIndex = 0
        while (groupIndex < numGroups) {
          val numColsInGroup = groupStartColumns(groupIndex + 1) - groupStartColumns(groupIndex)
          columnSetSizes(groupIndex) = Array.fill[Int](numColsInGroup)(0)
          columnSetIndices(groupIndex) =
            Array.fill[ArrayBuffer[Int]](numColsInGroup)(new ArrayBuffer[Int])
          columnSetValues(groupIndex) =
            Array.fill[ArrayBuffer[Double]](numColsInGroup)(new ArrayBuffer[Double])
          groupIndex += 1
        }
        iterator.foreach {
          case (dv: DenseVector, rowIndex: Long) =>
            var groupIndex = 0
            while (groupIndex < numGroups) {
              val fromCol = groupStartColumns(groupIndex)
              val numColsInGroup = groupStartColumns(groupIndex + 1) - groupStartColumns(groupIndex)
              var colIdx = 0
              while (colIdx < numColsInGroup) {
                columnSetSizes(groupIndex)(colIdx) += 1
                columnSetIndices(groupIndex)(colIdx) += rowIndex.toInt
                columnSetValues(groupIndex)(colIdx) += dv(fromCol + colIdx)
                colIdx += 1
              }
              groupIndex += 1
            }
          case (sv: SparseVector, rowIndex: Long) =>
            /*
              A sparse vector is chopped into groups (destination partitions).
              We iterate through the non-zeros (indexed by k), going to the next group sv.indices(k)
              passes the current group's boundary.
             */
            var groupIndex = 0
            var k = 0 // index into SparseVector non-zeros
            val nnz = sv.indices.length
            while (groupIndex < numGroups && k < nnz) {
              val fromColumn = groupStartColumns(groupIndex)
              val groupEndColumn = groupStartColumns(groupIndex + 1)
              while (k < nnz && sv.indices(k) < groupEndColumn) {
                val columnIndex = sv.indices(k) // index in full row
                val colIdx = columnIndex - fromColumn // index in group of columns
                columnSetSizes(groupIndex)(colIdx) += 1
                columnSetIndices(groupIndex)(colIdx) += rowIndex.toInt
                columnSetValues(groupIndex)(colIdx) += sv.values(k)
                k += 1
              }
              groupIndex += 1
            }
        }
        Range(0, numGroups).map { groupIndex =>
          val numColsInGroup = groupStartColumns(groupIndex + 1) - groupStartColumns(groupIndex)
          val groupPartialColumns: Array[SparseVector] = Range(0, numColsInGroup).map { colIdx =>
            new SparseVector(columnSetSizes(groupIndex)(colIdx),
              columnSetIndices(groupIndex)(colIdx).toArray,
              columnSetValues(groupIndex)(colIdx).toArray)
          }.toArray
          (groupIndex, (sourcePartitionIndex, groupPartialColumns))
        }.toIterator
      }

    // Shuffle data
    val groupedPartialColumns: RDD[(Int, Iterable[(Int, Array[SparseVector])])] =
      partialColumns.groupByKey()

    // Each target partition now holds its set of columns.
    // Group the partial columns into full columns.
    val fullColumns = groupedPartialColumns.flatMap { case (groupIndex, iterable) =>
      val numColsInGroup = groupStartColumns(groupIndex + 1) - groupStartColumns(groupIndex)

      // We do not know the number of rows or non-zeros per group, so we need to collect the groups
      // before filling the full columns.
      // collectedPartCols(sourcePartitionIndex)(colIdx) = partial column
      val collectedPartCols = new Array[Array[SparseVector]](numSourcePartitions)
      // nzCounts(colIdx)(sourcePartitionIndex) = number of non-zeros
      val nzCounts = Array.fill[Array[Int]](numColsInGroup)(Array.fill[Int](numSourcePartitions)(0))
      val iter = iterable.iterator
      while (iter.hasNext) {
        val (sourcePartitionIndex, partCols) = iter.next()
        collectedPartCols(sourcePartitionIndex) = partCols
        var colIdx = 0
        while (colIdx < partCols.length) {
          val partCol = partCols(colIdx)
          nzCounts(colIdx)(sourcePartitionIndex) += partCol.indices.length
          colIdx += 1
        }
      }
      // nzOffsets(colIdx)(sourcePartitionIndex) = cumulative number of non-zeros
      val nzOffsets: Array[Array[Int]] = nzCounts.map(_.scanLeft(0)(_ + _))

      // Initialize full columns
      val columnNZIndices: Array[Array[Int]] =
        nzOffsets.map(colNZOffsets => new Array[Int](colNZOffsets.last))
      val columnNZValues: Array[Array[Double]] =
        nzOffsets.map(colNZOffsets => new Array[Double](colNZOffsets.last))

      // Fill columns
      var colIdx = 0 // index within group
      while (colIdx < numColsInGroup) {
        var sourcePartitionIndex = 0
        while (sourcePartitionIndex < numSourcePartitions) {
          val nzStartOffset = nzOffsets(colIdx)(sourcePartitionIndex)
          val partColLength = nzOffsets(colIdx)(sourcePartitionIndex + 1) - nzStartOffset
          Array.copy(collectedPartCols(sourcePartitionIndex)(colIdx).indices, 0,
            columnNZIndices(colIdx), nzStartOffset, partColLength)
          Array.copy(collectedPartCols(sourcePartitionIndex)(colIdx).values, 0,
            columnNZValues(colIdx), nzStartOffset, partColLength)
          sourcePartitionIndex += 1
        }
        colIdx += 1
      }
      val columns = columnNZIndices.zip(columnNZValues).map { case (indices, values) =>
        Vectors.sparse(numRows, indices, values)
      }
      val fromColumn = groupStartColumns(groupIndex)
      val columnIndices = Range(0, numColsInGroup).map(_ + fromColumn)
      columnIndices.zip(columns)
    }

    fullColumns
  }
}
