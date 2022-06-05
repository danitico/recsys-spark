package recommender.collaborative.user_based

import org.apache.spark.ml.linalg.Vector
import recommender.collaborative.BaseRecommender


class UserBasedRatingRecommender(kUsers: Int) extends BaseRecommender {
  protected var _kUsers: Int = kUsers

  def setKUsers(k: Int): Unit = {
    this._kUsers = k
  }

  protected def getKSimilarUsers(targetUser: Array[Double], item: Int): List[(Double, Vector, Double)] = {
    val usersWithRating = this._matrix.rowIter.filter(_(item) > 0).toList

    if (usersWithRating.isEmpty) {
      return List()
    }

    val meanRatingUsers = usersWithRating.map(f => {
      val ratedItems = f.toArray.filter(_ > 0)

      ratedItems.sum / ratedItems.length
    })

    val correlations = usersWithRating.map(
      f => this._similarity.getSimilarity(targetUser, f.toArray)
    )

    correlations.zip(usersWithRating).zip(meanRatingUsers).map{
      case ((a, b), c) => (a, b, c)
    }.sortWith(_._1 > _._1).take(this._kUsers)
  }

  protected def ratingCalculation(topKUsers: List[(Double, Vector, Double)], ratingMean: Double, item: Int): Double = {
    val numerator = topKUsers.map(a => {
      a._1 * (a._2(item) - a._3)
    }).sum

    val denominator = topKUsers.map(_._1).sum

    ratingMean + (numerator/denominator)
  }

  def predictionRatingItem(targetUser: Array[Double], item: Int): Double = {
    val ratedItems = targetUser.filter(_ > 0)
    val ratingMean = ratedItems.sum / ratedItems.length

    val topKUsers = this.getKSimilarUsers(targetUser, item - 1)

    if (topKUsers.isEmpty) {
      return 0.0
    }

    this.ratingCalculation(topKUsers, ratingMean, item - 1)
  }
}
