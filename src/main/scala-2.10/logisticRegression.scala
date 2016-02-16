/**
 * Created by lkhamsurenl on 2/16/16.
 */
/**
 * Created by lkhamsurenl on 2/15/16.
 */

import java.io.File

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.util.MLUtils

import scala.io.Source

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.rdd._

object logisticRegression {
  /*
  * Reading data from RDD[String] to generate RDD[LabeledPoint]
  * */
  def createLabeledPoints(data: RDD[String]): RDD[LabeledPoint] = {
    data.map { line =>
      val fields = line.split(",")
      // format: LabeledPoint(label, Vector(feature_0, feature_1))
      LabeledPoint(fields(2).toInt, Vectors.dense(fields(0).toDouble, fields(1).toDouble))
    }
  }

  def main(args: Array[String]): Unit = {

    // Set up environment.
    val conf = new SparkConf()
      .setAppName("MovieRecommendation")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    // Read data.
    val data = createLabeledPoints(sc.textFile(new File("resources/logistic_regression.txt").toString()))

    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(x => x.features))

    val zipped_data = data.map(l => LabeledPoint(l.label, scaler2.transform(l.features))).zipWithIndex()
    val training_data = zipped_data.filter{case (l:LabeledPoint, i: Long) => i % 10 <= 7}.map{case (l, i) => l}
    val test_data = zipped_data.filter{case (l:LabeledPoint, i: Long) => i % 10 > 7}.map{case (l, i) => l}

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training_data)

    val valuesAndPreds = test_data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(valuesAndPreds)
    val precision = metrics.precision // % of the test it got correct.
    println("Precision = " + precision)

  }
}

