/**
 * Created by lkhamsurenl on 2/15/16.
 */

import java.io.File

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.util.MLUtils

import scala.io.Source

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.rdd._

object predictEarthquake {
  /*
  * Reading data from RDD[String] to generate RDD[LabeledPoint]
  * */
  def createLabeledPoints(data: RDD[String]): RDD[LabeledPoint] = {
      data.filter(line => !line.contains("@")) // Remove the headers.
      .map { line =>
      val fields = line.split(",")
      // format: LabeledPoint(Magnitude, Vector(focal_depth, lat, long))
      LabeledPoint(fields(3).toDouble, Vectors.dense(fields(0).toDouble, fields(1).toDouble, fields(2).toDouble))
    }
  }

  def main(args: Array[String]): Unit = {
    val quakeDir = "resources/quake-5-fold"

    // Set up environment.
    val conf = new SparkConf()
    .setAppName("MovieRecommendation")
    .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    // Read data.
    val training_data_u = (1 to 5).map(i =>
      createLabeledPoints(sc.textFile(new File(quakeDir, s"quake-5-${i}tra.dat").toString()))
    ).reduce((l1, l2) => l1.union(l2))

    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(training_data_u.map(x => x.features))

    val training_data = training_data_u.map(point => LabeledPoint(point.label, scaler2.transform(point.features)))
    val numIterations = 20
    val model = LinearRegressionWithSGD.train(training_data, numIterations)

    // Evaluate model on the test data.
    val test_data1 = createLabeledPoints(sc.textFile(new File(quakeDir, "quake-5-1tst.dat").toString()))

    val valuesAndPreds = test_data1.map { point =>
      val prediction = model.predict(scaler2.transform(point.features))
      (point.label, prediction)
    }

    val pres = valuesAndPreds.collect()
    println(s"predictions:")
    pres.foreach(println)
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("training Mean Squared Error = " + MSE)
  }
}
