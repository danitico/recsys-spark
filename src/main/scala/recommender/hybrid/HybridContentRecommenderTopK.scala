package recommender.hybrid

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import recommender.content.ContentBasedTopKRecommender
import recommender.sequential.SequentialTopKRecommender

class HybridContentRecommenderTopK extends Serializable {
  var _contentFiltering: ContentBasedTopKRecommender = null
  var _sequential: SequentialTopKRecommender = null
  var _numberOfItems: Int = 0
  var _k: Int = 5
  var _weightCf: Double = 0.6
  var _weightSequential: Double = 0.4


  def setNumberOfItems(numberOfItems: Int): this.type = {
    this._numberOfItems = numberOfItems
    this
  }

  def setCF(recSys: ContentBasedTopKRecommender): this.type = {
    this._contentFiltering = recSys
    this
  }

  def setSequential(recSys: SequentialTopKRecommender): this.type = {
    this._sequential = recSys
    this
  }

  def fit(train: DataFrame): Unit = {
    this._sequential.fit(train)

    this._contentFiltering.fit(
      train, this._numberOfItems
    )
  }

  def transform(test: DataFrame): (Seq[(Int, Double)], Seq[(Int, Double)]) = {
    val predictionsSequential = this._sequential.transform(
      test
    ).map(element => {
      (element._1, element._2 * this._weightSequential)
    })

    val explicitArray = Vectors.sparse(
      this._numberOfItems,
      test.select("item_id", "rating").collect().map(row => {
        (row.getInt(0) - 1, row.getDouble(1))
      })
    ).toDense.toArray

    val predictionsCF = this._contentFiltering.transform(
      explicitArray
    )

    val maxValueScore = predictionsCF.head._2

    val normalizedPredictionsCF = predictionsCF.map(element => {
      (element._1, (element._2 / maxValueScore) * this._weightCf)
    })

    val combination = (predictionsSequential ++ normalizedPredictionsCF).groupBy(_._1).mapValues(
      _.map(_._2).sum
    ).toArray

    (normalizedPredictionsCF, combination.sortWith(_._2 > _._2).take(this._k).toSeq)
  }
}
