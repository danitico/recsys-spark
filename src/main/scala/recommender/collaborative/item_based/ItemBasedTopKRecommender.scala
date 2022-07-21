package recommender.collaborative.item_based

import org.apache.spark.ml.linalg.Vector
import recommender.collaborative.BaseRecommender

import scala.math.abs


class ItemBasedTopKRecommender(kSimilarItems: Int, kRecommendedItems: Int) extends BaseRecommender(isUserBased = false){
  protected var _kSimilarItems: Int = kSimilarItems
  protected var _kRecommendedItems: Int = kRecommendedItems

  def setNumberSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  def setKRecommendedItems(k: Int): Unit = {
    this._kRecommendedItems = k
  }

  protected def getKSimilarItems(targetItem: Array[Double], user: Int): List[(Double, Vector)] = {
    val itemsWithRating = this._matrix.rowIter.filter(_(user) > 0).toList

    if (itemsWithRating.isEmpty) {
      return List()
    }

    val correlations = itemsWithRating.map(
      f => this._similarity.getSimilarity(targetItem, f.toArray)
    )

    if (correlations.forall(_.isNaN)) {
      return List()
    }

    correlations.zip(itemsWithRating).sortWith(_._1 > _._1).take(this._kSimilarItems)
  }

  protected def ratingCalculation(topKItems: List[(Double, Vector)], user: Int): Double = {
    val numerator = topKItems.map(a => {
      a._1 * a._2(user)
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    numerator/denominator
  }

  def topKItemsForUser(targetUser: Array[Double], user: Int): Set[Int] = {
    val unratedItems = targetUser.zipWithIndex.filter(_._1 == 0).map(_._2)
    val itemsRatings = this._matrix.rowIter.zipWithIndex.filter(f => {
      unratedItems.contains(f._2)
    }).toList

    itemsRatings.map(g => {
      val similarItems = this.getKSimilarItems(g._1.toArray, user - 1)

      if (similarItems.isEmpty) {
        (g._2 + 1, 0.0)
      } else {
        val rating = this.ratingCalculation(similarItems, user - 1)

        (g._2 + 1, rating)
      }
    }).toArray.sortWith(_._2 > _._2).map(_._1).take(this._kRecommendedItems).toSet
  }
}
