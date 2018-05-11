name := "Yggdrasil"

version := "1.0.1"

scalaVersion := "2.10.4"

spName := "fabuzaid21/yggdrasil" 

sparkVersion := "1.6.0" 

sparkComponents ++= Seq("mllib", "sql")

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"

spShortDescription := "Yggdrasil: Faster Decision Trees Using Column Partitioning in Spark"

spDescription := """Yggdrasil is a more efficient way in [Apache Spark](http://spark.apache.org)
                    | to train decision trees for large depths and datasets with a
                    | high number of features. For depths greater than 10, Yggdrasil is an order
                    | of magnitude faster than Spark MLlib v1.6.0.""".stripMargin

// You must have an Open Source License. Some common licenses can be found in: http://opensource.org/licenses
licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")