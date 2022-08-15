/*
  recommender/collaborative/user_based/UserBasedTopKRecommender.scala
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

import recommender.BaseRecommender


class UserBasedTopKRecommender(kUsers: Int, kItems: Int, numberOfItems: Long) extends BaseRecommender(numberOfItems = numberOfItems, isUserBased = true) {
  private var _kUsers: Int = kUsers
  private var _kItems: Int = kItems
  private var _candidates: List[Array[Double]] = null

  def getKUsers: Int = this._kUsers

  def setKUsers(k: Int): Unit = {
    this._kUsers = k
  }

  def getKItems: Int = this._kItems

  def setKItems(k: Int): Unit = {
    this._kItems = k
  }

  private def getKSimilarUsers(targetUser: Array[Double], item: Int): List[(Double, Array[Double], Double)] = {
    val usersWithRating = this._candidates.filter(_(item) > 0)

    if (usersWithRating.isEmpty) {
      return List()
    }

    val meanRatingUsers = usersWithRating.map(f => {
      val ratedItems = f.filter(_ > 0)

      ratedItems.sum / ratedItems.length
    })

    val correlations = usersWithRating.map(
      f => this._similarity.getSimilarity(targetUser, f)
    )

    correlations.zip(usersWithRating).zip(meanRatingUsers).map {
      case ((a, b), c) => (a, b, c)
    }.sortWith(_._1 > _._1).take(this._kUsers)
  }

  private def ratingCalculation(topKUsers: List[(Double, Array[Double], Double)], ratingMean: Double, item: Int): Double = {
    val numerator = topKUsers.map(a => {
      a._1 * (a._2(item) - a._3)
    }).sum

    val denominator = topKUsers.map(_._1).sum

    ratingMean + (numerator/denominator)
  }

  override def transform(targetUser: Array[Double]): Seq[(Int, Double)] = {
    val ratedItems = targetUser.filter(_ > 0)
    val ratingMean = ratedItems.sum / ratedItems.length

    this._candidates = this._matrix.rowIter.map(_.toArray).toList

    val unratedItems = targetUser.zipWithIndex.filter(_._1 == 0).map(_._2)

    unratedItems.map(item => {
      val similarUsers = this.getKSimilarUsers(targetUser, item)

      if (similarUsers.isEmpty) {
        (item + 1, 0.0)
      } else {
        val rating = this.ratingCalculation(similarUsers, ratingMean, item)

        (item + 1, rating)
      }
    }).sortBy(- _._2).take(this._kItems).toSeq
  }
}
