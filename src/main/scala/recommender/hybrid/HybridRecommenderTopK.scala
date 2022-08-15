/*
  recommender/hybrid/HybridRecommenderTopK.scala
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
package recommender.hybrid

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame

import recommender.BaseRecommender
import recommender.sequential.SequentialTopKRecommender


class HybridRecommenderTopK(kRecommendedItems: Int, numberOfItems: Long) extends BaseRecommender(numberOfItems = numberOfItems) {
  private var _k: Int = kRecommendedItems
  private var _firstRecommender: BaseRecommender = null
  private var _isFirstRecommenderSequential = false
  private var _secondRecommender: BaseRecommender = null
  private var _isSecondRecommenderSequential = false
  private var _weightFirstRecommender: Double = 0.6
  private var _weightSecondRecommender: Double = 0.4

  def getKRecommendedItems: Int = this._k

  def setKRecommendedItems(kRecommendedItems: Int): this.type = {
    this._k = kRecommendedItems
    this
  }

  def getWeightFirstRecommender: Double = this._weightFirstRecommender

  def setWeightFirstRecommender(weight: Double): this.type = {
    this._weightFirstRecommender = weight
    this
  }

  def getWeightSecondRecommender: Double = this._weightSecondRecommender

  def setWeightSecondRecommender(weight: Double): this.type = {
    this._weightSecondRecommender = weight
    this
  }

  def setFirstRecommender(recSys: BaseRecommender): this.type = {
    this._firstRecommender = recSys
    this._isFirstRecommenderSequential = recSys.isInstanceOf[SequentialTopKRecommender]
    this
  }

  def setSecondRecommender(recSys: BaseRecommender): this.type = {
    this._secondRecommender = recSys
    this._isSecondRecommenderSequential = recSys.isInstanceOf[SequentialTopKRecommender]
    this
  }

  private def normalizeRanking(ranking: Seq[(Int, Double)], weight: Double): Seq[(Int, Double)] = {
    if (ranking.isEmpty) {
      Seq()
    } else {
      val maxValue = ranking.head._2

      ranking.map(element => {
        (element._1, (element._2 / maxValue) * weight)
      })
    }
  }

  override def fit(train: DataFrame): Unit = {
    this._firstRecommender.fit(train)
    this._secondRecommender.fit(train)
  }

  override def transform(test: DataFrame): Seq[(Int, Double)] = {
    var predictionsFirstRecommender: Seq[(Int, Double)] = Seq()
    var predictionsSecondRecommender: Seq[(Int, Double)] = Seq()

    val explicitArray = Vectors.sparse(
      this._numberOfItems.toInt,
      test.select("item_id", "rating").collect().map(row => {
        (row.getInt(0) - 1, row.getDouble(1))
      })
    ).toDense.toArray

    if (this._isFirstRecommenderSequential) {
      predictionsFirstRecommender = this._firstRecommender.transform(test)
    } else {
      predictionsFirstRecommender = this._firstRecommender.transform(explicitArray)
    }

    if (this._isSecondRecommenderSequential) {
      predictionsSecondRecommender = this._secondRecommender.transform(test)
    } else {
      predictionsSecondRecommender = this._secondRecommender.transform(explicitArray)
    }

    val normalizedPredictionsFirstRecommender = this.normalizeRanking(
      predictionsFirstRecommender, this._weightFirstRecommender
    )
    val normalizedPredictionsSecondRecommender = this.normalizeRanking(
      predictionsSecondRecommender, this._weightSecondRecommender
    )

    val combination = (normalizedPredictionsFirstRecommender ++ normalizedPredictionsSecondRecommender).groupBy(_._1).mapValues(
      _.map(_._2).sum
    ).toArray

    combination.sortWith(_._2 > _._2).take(this._k).toSeq
  }
}
