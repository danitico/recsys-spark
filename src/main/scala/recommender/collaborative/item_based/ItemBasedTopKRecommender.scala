package recommender.collaborative.item_based

import scala.math.abs

import org.apache.spark.ml.linalg.Vector

import recommender.BaseRecommender


class ItemBasedTopKRecommender(kSimilarItems: Int, kRecommendedItems: Int, numberOfItems: Long) extends BaseRecommender(numberOfItems, isUserBased = false){
  private var _kSimilarItems: Int = kSimilarItems
  private var _kRecommendedItems: Int = kRecommendedItems
  private var _ratingsOfItemsRatedByUser: List[(Vector, Int)] = null

  def getNumberSimilarItems: Int = this._kSimilarItems

  def setNumberSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  def getKRecommendedItems: Int = this._kRecommendedItems

  def setKRecommendedItems(k: Int): Unit = {
    this._kRecommendedItems = k
  }

  private def getKSimilarItems(targetItem: Array[Double]): List[(Double, Int)] = {
    if (this._ratingsOfItemsRatedByUser.isEmpty) {
      return List()
    }

    val correlations = this._ratingsOfItemsRatedByUser.map(
      f => this._similarity.getSimilarity(targetItem, f._1.toArray)
    )

    if (correlations.forall(_.isNaN)) {
      return List()
    }

    correlations.zip(this._ratingsOfItemsRatedByUser).map{
      case (a, (b, c)) => (a, c)
    }.sortWith(_._1 > _._1).take(this._kSimilarItems)
  }

  private def ratingCalculation(topKItems: List[(Double, Int)], targetUser: Array[Double]): Double = {
    val numerator = topKItems.map(a => {
      a._1 * targetUser(a._2)
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    numerator/denominator
  }

  override def transform(targetUser: Array[Double]): Seq[(Int, Double)] = {
    val unratedItems = targetUser.zipWithIndex.filter(_._1 == 0).map(_._2)

    this._ratingsOfItemsRatedByUser = this._matrix.rowIter.toList.zipWithIndex.filter(f => {
      !unratedItems.contains(f._2)
    }).map(f => (f._1, f._2 + 1))

    val ratingsOfItemsNotRatedByUser = this._matrix.rowIter.toList.zipWithIndex.filter(f => {
      unratedItems.contains(f._2)
    }).map(f => (f._1, f._2 + 1))

    ratingsOfItemsNotRatedByUser.map(g => {
      val similarItems = this.getKSimilarItems(g._1.toArray)

      if (similarItems.isEmpty) {
        (g._2, 0.0)
      } else {
        val rating = this.ratingCalculation(similarItems, targetUser)

        (g._2, rating)
      }
    }).toArray.sortBy(- _._2).take(this._kRecommendedItems).toSeq
  }
}
