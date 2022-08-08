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
