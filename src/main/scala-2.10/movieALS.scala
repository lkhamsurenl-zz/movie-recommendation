/**
 * Created by lkhamsurenl on 1/2/16.
 */

import java.io.File

import scala.io.Source

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object movieALS {
  def main(args: Array[String]): Unit = {
    // Set up the logger for better debugging.
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length != 2) {
      println(s"Must have exactly 2 arguments. Given: ${args.length}")
      System.exit(-1)
    }

    // Set up environment.
    val conf = new SparkConf()
      .setAppName("MovieRecommendation")
      .set("spark.executor.memory", "2g")
    val sc = new SparkContext(conf)

    // Load personal rating and create RDD.
    val personalRating = loadRating(args(0))
    val personalRatingsRDD = sc.parallelize(personalRating, 1)

    // Load ratings and movie titles.
    val movieLensHomeDir = args(1)
    val ratings = sc.textFile(new File(movieLensHomeDir, "ratings.dat").toString).map { line =>
      val fields = line.split("::")
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }

    val movies = sc.textFile(new File(movieLensHomeDir, "movies.dat").toString).map { line =>
      val fields = line.split("::")
      // format: Map(movieId -> movieName)
      (fields(0).toInt, fields(1))
    }.collect().toMap

    val numPartition = 4
    // 60% of the data is training data.
    val trainingData = ratings.filter(fields => fields._1 <= 6)
      .values
      .union(personalRatingsRDD)
      .repartition(numPartition)
      .cache()

    // 20% of data is test.
    val testData = ratings.filter(fields => fields._1 > 6 && fields._1 <= 8)
      .values
      .cache()

    // Remaining 20% is validation.
    val validationData = ratings.filter(fields => fields._1 > 8)
      .values
      .repartition(numPartition)
      .cache()

    val bestModel = Model.run(trainingData, validationData)

    val testRMSE = Model.computeRmse(bestModel.matrix.get, testData, testData.count)

    val predictionData = ratings.map{ rating =>
      (0,rating._2.product)
    }.distinct()
    val predictions = bestModel.matrix.get
      .predict(predictionData)
      .collect()
      .sortBy(- _.rating)
      .take(50)
      .foreach { prediction =>
        println(s"${prediction.product}: ${movies(prediction.product)}")
      }

    //cleanup:
    sc.stop()
  }

  def loadRating(path: String): Seq[Rating] = {
    val lines = Source.fromFile(path).getLines()
    val ratings = lines.map {line =>
      val fields = line.split("::")
      // userID, productID, weight
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }.filter(_.rating > 0.0)
    if (ratings.isEmpty) {
      sys.error("No ratings provided")
    } else {
      ratings.toSeq
    }
  }
}
