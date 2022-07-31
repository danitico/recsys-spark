package recommender.collaborative.explicit.item_based

import org.apache.spark.ml.linalg.Vector
import recommender.collaborative.explicit.ExplicitBaseRecommender

import scala.math.abs


class ItemBasedTopKRecommender(kSimilarItems: Int, kRecommendedItems: Int) extends ExplicitBaseRecommender(isUserBased = false){
  protected var _kSimilarItems: Int = kSimilarItems
  protected var _kRecommendedItems: Int = kRecommendedItems
  protected var _ratingsOfItemsRatedByUser: List[(Vector, Int)] = null

  def setNumberSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  def setKRecommendedItems(k: Int): Unit = {
    this._kRecommendedItems = k
  }

  protected def getKSimilarItems(targetItem: Array[Double]): List[(Double, Int)] = {
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

  protected def ratingCalculation(topKItems: List[(Double, Int)], targetUser: Array[Double]): Double = {
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
