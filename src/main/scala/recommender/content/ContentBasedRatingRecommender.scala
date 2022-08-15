/*
  recommender/content/ContentBasedRatingRecommender.scala
  Copyright (C) 2022 Daniel Ranchal Parrado <danielranchal@correo.ugr.es>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>
*/
package recommender.content

import scala.math.abs

import similarity.EuclideanSimilarity


class ContentBasedRatingRecommender(kSimilarItems: Int, numberOfItems: Long) extends ContentRecommender(numberOfItems) {
  private var _kSimilarItems: Int = kSimilarItems

  def getKSimilarItems: Int = this._kSimilarItems

  def setKSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  private def solveSimilarity(targetItem: Array[Double], otherItem: Array[Double]): Double = {
    val similarity = this._similarity.getSimilarity(targetItem, otherItem)

    if (similarity == 0.0) {
      new EuclideanSimilarity().getSimilarity(targetItem, otherItem)
    } else {
      similarity
    }
  }

  private def getKSimilarItems(targetItem: Array[Double], user: Int): List[(Double, Double)] = {
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
