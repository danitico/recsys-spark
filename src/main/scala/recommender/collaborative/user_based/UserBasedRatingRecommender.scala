/*
  recommender/collaborative/user_based/UserBasedRatingRecommender.scala
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
package recommender.collaborative.user_based

import org.apache.spark.ml.linalg.Vector

import recommender.BaseRecommender


class UserBasedRatingRecommender(kUsers: Int, numberOfItems: Long) extends BaseRecommender(numberOfItems = numberOfItems, isUserBased = true) {
  private var _kUsers: Int = kUsers

  def getKUsers: Int = this._kUsers

  def setKUsers(k: Int): Unit = {
    this._kUsers = k
  }

  private def getKSimilarUsers(targetUser: Array[Double], item: Int): List[(Double, Vector, Double)] = {
    val usersWithRating = this._matrix.rowIter.filter(_(item) > 0).toList

    if (usersWithRating.isEmpty) {
      return List()
    }

    val meanRatingUsers = usersWithRating.map(f => {
      val ratedItems = f.toArray.filter(_ > 0)

      ratedItems.sum / ratedItems.length
    })

    val correlations = usersWithRating.map(
      f => this._similarity.getSimilarity(targetUser, f.toArray)
    )

    correlations.zip(usersWithRating).zip(meanRatingUsers).map{
      case ((a, b), c) => (a, b, c)
    }.sortWith(_._1 > _._1).take(this._kUsers)
  }

  private def ratingCalculation(topKUsers: List[(Double, Vector, Double)], ratingMean: Double, item: Int): Double = {
    val numerator = topKUsers.map(a => {
      a._1 * (a._2(item) - a._3)
    }).sum

    val denominator = topKUsers.map(_._1).sum

    ratingMean + (numerator/denominator)
  }

  override def transform(targetUser: Array[Double], item: Int): Double = {
    val ratedItems = targetUser.filter(_ > 0)
    val ratingMean = ratedItems.sum / ratedItems.length

    val topKUsers = this.getKSimilarUsers(targetUser, item - 1)

    if (topKUsers.isEmpty) {
      return 0.0
    }

    this.ratingCalculation(topKUsers, ratingMean, item - 1)
  }
}
