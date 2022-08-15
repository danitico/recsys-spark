/*
  recommender/collaborative/item_based/ItemBasedRatingRecommender.scala
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
package recommender.collaborative.item_based

import scala.math.abs

import org.apache.spark.ml.linalg.Vector

import recommender.BaseRecommender


class ItemBasedRatingRecommender(kSimilarItems: Int, numberOfItems: Long) extends BaseRecommender(numberOfItems, isUserBased = false){
  private var _kSimilarItems: Int = kSimilarItems

  def getNumberSimilarItems: Int = this._kSimilarItems

  def setNumberSimilarItems(k: Int): Unit = {
    this._kSimilarItems = k
  }

  private def getKSimilarItems(targetItem: Array[Double], user: Int): List[(Double, Vector)] = {
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

  private def ratingCalculation(topKItems: List[(Double, Vector)], user: Int): Double = {
    val numerator = topKItems.map(a => {
      a._1 * a._2(user)
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    numerator/denominator
  }

  override def transform(targetItem: Array[Double], user: Int): Double = {
    val topKItems = this.getKSimilarItems(targetItem, user - 1)

    if (topKItems.isEmpty) {
      return 0.0
    }

    this.ratingCalculation(topKItems, user - 1)
  }
}
