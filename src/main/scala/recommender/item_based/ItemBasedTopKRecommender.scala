package recommender.item_based

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import recommender.BaseRecommender

import scala.math.abs


class ItemBasedTopKRecommender(session: SparkSession) extends BaseRecommender(session, isUserBased = false){
  protected var _kSimilarItems: Int = -1
  protected var _kRecommendedItems: Int = -1

  def this(session: SparkSession, kSimilarItems: Int, kRecommendedItems: Int) = {
    this(session)
    this.setNumberSimilarItems(kSimilarItems)
    this.setKRecommendedItems(kRecommendedItems)
  }

  def setNumberSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  def setKRecommendedItems(k: Int): Unit = {
    this._kRecommendedItems = k
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
  //java.lang.IllegalArgumentException: Comparison method violates its general contract!
/*
  def topKItemsForUser(targetUser: Array[Double], user: Int): List[(Int, Double)] = {
    val unratedItems = targetUser.zipWithIndex.filter(_._1 == 0).map(_._2)

    this._matrix.rowIter.zipWithIndex.filter((f: (Vector, Int)) => {
      unratedItems.contains(f._2)
    }).map((g: (Vector, Int)) => {
      val similarItems = this.getKSimilarItems(g._1.toArray, user - 1)
      val rating = this.ratingCalculation(similarItems, user - 1)

      (g._2 + 1, rating)
    }).toArray.sortWith((a: (Int, Double), b: (Int, Double)) => {
      a._2 >= b._2
    }).take(this._kRecommendedItems).toList
  }
*/
}
