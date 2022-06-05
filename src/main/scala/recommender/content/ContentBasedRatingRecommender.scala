package recommender.content

import org.apache.spark.ml.linalg.SparseVector
import scala.math.abs


class ContentBasedRatingRecommender(kSimilarItems: Int) extends BaseRecommender {
  protected var _kSimilarItems: Int = kSimilarItems

  def setNumberSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  protected def getKSimilarItems(targetItem: Array[Double], user: Int): List[(Double, Double)] = {
    val itemsWithRating = this._matrix.rowIter.zipWithIndex.filter(
      _._1(user) > 0
    ).map(tuple => (tuple._1(user), tuple._2 + 1)).toList

    if (itemsWithRating.isEmpty) {
      return List()
    }

    val correlations = itemsWithRating.map(tuple => {
      val itemFeature = this._features.filter(_.getInt(0) == tuple._2).head.getAs[SparseVector](1)
      this._similarity.getSimilarity(targetItem, itemFeature.toDense.toArray)
    })

    if (correlations.forall(_.isNaN)) {
      return List()
    }

    correlations.zip(itemsWithRating).map{
      case (a, (b, c)) => (a, b)
    }.sortWith(_._1 > _._1).take(this._kSimilarItems)
  }

  protected def ratingCalculation(topKItems: List[(Double, Double)]): Double = {
    val numerator = topKItems.map(a => {
      a._1 * a._2
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    numerator/denominator
  }

  def predictionRatingItem(targetItem: Array[Double], user: Int): Double = {
    val topKItems = this.getKSimilarItems(targetItem, user - 1)

    if (topKItems.isEmpty) {
      return 0.0
    }

    this.ratingCalculation(topKItems)
  }
}
