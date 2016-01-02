/**
 * Created by lkhamsurenl on 1/2/16.
 */

import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._

case class Model(val matrix: Option[MatrixFactorizationModel],
                 val rank: Int,
                 val iteration: Int,
                 val lambda: Double,
                 val validationRMSE: Double)

object Model {
  val ranks = List(4,8,12)
  val iterations = List(10, 20)
  val lambdas = List(0.1, 10.0)

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }

  def run(trainingData: RDD[Rating], validationData: RDD[Rating]): Model = {
    val numValidation = validationData.count()
    var bestModel: Model = Model(None, ranks(0), iterations(0), lambdas(0), Double.MaxValue)
    for {
      rank   <- ranks
      it     <- iterations
      lambda <- lambdas
    } {
      val matrix = ALS.train(trainingData, rank, it, lambda)
      val rmse = computeRmse(matrix, validationData, numValidation)
      if (rmse < bestModel.validationRMSE) {
        bestModel = Model(Some(matrix), rank, it, lambda, rmse)
      }
    }
    bestModel
  }
}
