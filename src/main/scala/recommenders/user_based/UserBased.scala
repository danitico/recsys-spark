package recommenders.user_based

import scala.math.abs

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession

import recommenders.BaseRecommender


class UserBased(session: SparkSession) extends BaseRecommender(session) {
  def getKSimilarUsers(targetUser: Array[Double], k: Int, item: Int): List[(Double, Vector)] = {
    val usersWithRating = this.matrix.rowIter.filter(_(item) > 0).toList

    val correlations = usersWithRating.map(
      f => this.similarity.getSimilarity(targetUser, f.toArray)
    )

    correlations.zip(usersWithRating).sortWith(_._1 > _._1).take(k)
  }

  def ratingCalculation(topKUsers: List[(Double, Vector)], ratingMean: Double, item: Int): Double = {
    val numerator = topKUsers.map(a => {
      a._1 * a._2(item)
    }).sum

    val denominator = topKUsers.map(_._1).reduce(abs(_) + abs(_))

    ratingMean + (numerator/denominator)
  }

  def predictionRatingItem(targetUser: Array[Double], item: Int): Double = {
    val topKUsers = this.getKSimilarUsers(targetUser, 25, item - 1)
    val ratingMean = targetUser.sum / targetUser.length

    this.ratingCalculation(topKUsers, ratingMean, item - 1)
  }

  def topKItemsForUser(targetUser: Array[Double], k: Int): List[(Int, Double)] = {
    val unratedItems = targetUser.zipWithIndex.filter(_._1 == 0).map(_._2)

    unratedItems.map(item => {
      (item, this.predictionRatingItem(targetUser, item))
    }).sortWith(_._2 > _._2).take(k).toList
  }
}
