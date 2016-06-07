# Yggdrasil: Faster Decision Trees Using Column Partitioning in Spark

Yggdrasil is a more efficient way in [Apache Spark](http://spark.apache.org) to
train decision trees for large depths and datasets with a high number of
features. For depths greater than 10, Yggdrasil is an order of magnitude faster
than Spark MLlib v1.6.0.

## Usage

Add the dependency to your SBT project by adding the following to `build.sbt`
(see the [Spark Packages
listing](http://spark-packages.org/package/amplab/spark-indexedrdd) for
spark-submit and Maven instructions):

```scala
resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

libraryDependencies += "fabuzaid21" % "yggdrasil" % "1.0"
```

Then use Yggdrasil as follows:

```scala
import org.apache.spark.ml.tree.impl.Yggdrasil

// Identical to the Spark MLlib Decision Tree API
val dt = new YggdrasilClassifier()
      .setFeaturesCol("indexedFeatures")
      .setLabelCol(labelColName)
      .setMaxDepth(params.maxDepth)
      .setMaxBins(params.maxBins)
      .setMinInstancesPerNode(params.minInstancesPerNode)
      .setMinInfoGain(params.minInfoGain)
      .setCacheNodeIds(params.cacheNodeIds)
      .setCheckpointInterval(params.checkpointInterval)
```
