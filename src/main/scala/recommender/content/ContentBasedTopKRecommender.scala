package recommender.content

import org.apache.spark.ml.linalg.SparseVector
import similarity.EuclideanSimilarity

import scala.math.abs

class ContentBasedTopKRecommender(kSimilarItems: Int) extends BaseRecommender {
  protected var _kSimilarItems: Int = kSimilarItems

  def setNumberSimilarItems(k: Int): Unit = {
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
    ).map(tuple => (tuple._1(user), tuple._2 + 1)).toList

    if (itemsWithRating.isEmpty) {
      return List()
    }

    val correlations = itemsWithRating.map(tuple => {
      val itemFeature = this._features.filter(_.getInt(0) == tuple._2).head.getAs[SparseVector](1)
      this.solveSimilarity(targetItem, itemFeature.toDense.toArray)
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
    }).toArray.sortBy(- _._2).take(this._kRecommendedItems).toList

    val topKItems = this.getKSimilarItems(targetItem, user - 1)

    if (topKItems.isEmpty) {
      return 0.0
    }

    this.ratingCalculation(topKItems)
  }
}
