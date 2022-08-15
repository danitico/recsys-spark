/*
  recommender/content/ContentRecommender.scala
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

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.DataFrame

import recommender.BaseRecommender


class ContentRecommender(numberOfItems: Long) extends BaseRecommender(numberOfItems = numberOfItems, isUserBased = false) {
  protected var _features: List[(Int, Array[Double])] = null

  def getFeatures: List[(Int, Array[Double])] = this._features

  def setFeatures(features: DataFrame): Unit = {
    this._features = this.transformFeatures(features)
  }

  private def transformFeatures(features: DataFrame): List[(Int, Array[Double])] = {
    val assembler = new VectorAssembler()
    val columnsToTransform = features.columns.drop(1)

    assembler.setInputCols(
      columnsToTransform
    ).setOutputCol("features")

    val transformed = assembler.transform(
      features
    ).drop(
      columnsToTransform:_*
    )

    transformed.collect().map(row => {
      (row.getInt(0), row.getAs[SparseVector](1).toDense.toArray)
    }).toList
  }
}
