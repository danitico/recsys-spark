package recommender.item_based

import scala.math.abs

import org.apache.spark.ml.linalg.Vector

import recommender.BaseRecommender


class ItemBasedRatingRecommender(kSimilarItems: Int) extends BaseRecommender(isUserBased = false){
  protected var _kSimilarItems: Int = kSimilarItems

  def setNumberSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  protected def getKSimilarItems(targetItem: Array[Double], user: Int): List[(Double, Vector)] = {
    val itemsWithRating = this._matrix.rowIter.filter(_(user) > 0).toList

    val correlations = itemsWithRating.map(
      f => this._similarity.getSimilarity(targetItem, f.toArray)
    )

    correlations.zip(itemsWithRating).sortWith(_._1 > _._1).take(this._kSimilarItems)
  }

  protected def ratingCalculation(topKItems: List[(Double, Vector)], user: Int): Double = {
    val numerator = topKItems.map(a => {
      a._1 * a._2(user)
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    numerator/denominator
  }

  def predictionRatingItem(targetItem: Array[Double], user: Int): Double = {
    val topKItems = this.getKSimilarItems(targetItem, user - 1)

    this.ratingCalculation(topKItems, user - 1)
  }
}
