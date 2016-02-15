import AssemblyKeys._

assemblySettings

name := "MovieRecommendation"

version := "0.1"

scalaVersion := "2.10.3"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.3.0" % "provided"
    