package recommender.content

import scala.math.abs

import similarity.EuclideanSimilarity


class ContentBasedRatingRecommender(kSimilarItems: Int, numberOfItems: Long) extends ContentRecommender(numberOfItems) {
  protected var _kSimilarItems: Int = kSimilarItems

  def setKSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  protected def solveSimilarity(targetItem: Array[Double], otherItem: Array[Double]): Double = {
    val similarity = this._similarity.getSimilarity(targetItem, otherItem)

    if (similarity == 0.0) {
      new EuclideanSimilarity().getSimilarity(targetItem, otherItem)
    } else {
      similarity
    }
  }

  protected def getKSimilarItems(targetItem: Array[Double], user: Int): List[(Double, Double)] = {
    val itemsWithRating = this._matrix.rowIter.zipWithIndex.filter(
      _._1(user) > 0
    ).map(tuple => {
      (tuple._1(user), tuple._2 + 1)
    }).toList

    if (itemsWithRating.isEmpty) {
      return List()
    }

    val correlations = itemsWithRating.map(tuple => {
      val itemFeature = this._features.filter(_._1 == tuple._2).head._2
      this.solveSimilarity(targetItem, itemFeature)
    })

    if (correlations.forall(_.isNaN)) {
      return List()
    }

    correlations.zip(itemsWithRating).map{
      case (a, (b, c)) => (a, b)
    }.sortWith(_._1 > _._1).take(this._kSimilarItems)
  }

  private def ratingCalculation(topKItems: List[(Double, Double)]): Double = {
    val numerator = topKItems.map(a => {
      a._1 * a._2
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    numerator/denominator
  }

  override def transform(targetItem: Array[Double], user: Int): Double = {
    val topKItems = this.getKSimilarItems(targetItem, user - 1)

    if (topKItems.isEmpty) {
      return 0.0
    }

    this.ratingCalculation(topKItems)
  }
}
