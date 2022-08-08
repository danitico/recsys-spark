package recommender.content

import scala.math.abs

import similarity.EuclideanSimilarity


class ContentBasedTopKRecommender(kSimilarItems: Int, kRecommendedItems: Int, numberOfItems: Long) extends ContentRecommender(numberOfItems) {
  protected var _kSimilarItems: Int = kSimilarItems
  protected var _kRecommendedItems: Int = kRecommendedItems
  protected var _itemsRatedByUser: List[Int] = null

  def setKSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  def setKRecommendedItems(k: Int): Unit = {
    this._kRecommendedItems = k
  }

  protected def solveSimilarity(targetItem: Array[Double], otherItem: Array[Double]): Double = {
    val similarity = this._similarity.getSimilarity(targetItem, otherItem)

    if (similarity == 0.0) {
      new EuclideanSimilarity().getSimilarity(targetItem, otherItem)
    } else {
      similarity
    }
  }

  protected def getKSimilarItems(targetItem: Array[Double]): List[(Double, Int)] = {
    if (this._itemsRatedByUser.isEmpty) {
      return List()
    }

    val correlations = this._itemsRatedByUser.map(
      f => {
        val itemFeature = this._features.filter(_._1 == f).head._2
        this.solveSimilarity(targetItem, itemFeature)
      }
    )

    if (correlations.forall(_.isNaN)) {
      return List()
    }

    correlations.zip(this._itemsRatedByUser).sortWith(_._1 > _._1).take(this._kSimilarItems)
  }

  private def ratingCalculation(topKItems: List[(Double, Int)], targetUser: Array[Double]): Double = {
    val numerator = topKItems.map(a => {
      a._1 * targetUser(a._2)
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    numerator/denominator
  }

  override def transform(targetUser: Array[Double]): Seq[(Int, Double)] = {
    val partition = targetUser.zipWithIndex.partition(_._1 == 0)
    val unratedItems = partition._1.map(_._2 + 1).toList
    this._itemsRatedByUser = partition._2.map(_._2 + 1).toList

    unratedItems.map(g => {
      val itemFeature = this._features.filter(_._1 == g).head._2
      val similarItems = this.getKSimilarItems(itemFeature)

      if (similarItems.isEmpty) {
        (g, 0.0)
      } else {
        val rating = this.ratingCalculation(similarItems, targetUser)

        (g, rating)
      }
    }).toArray.sortBy(- _._2).take(this._kRecommendedItems).toSeq
  }
}
