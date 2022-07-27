package recommender.hybrid

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import recommender.collaborative.explicit.ExplicitBaseRecommender
import recommender.sequential.SequentialTopKRecommender

class HybridRecommenderTopK extends Serializable {
  var collaborativeFiltering: ExplicitBaseRecommender = null
  var sequential: SequentialTopKRecommender = null
  var numberOfItems: Int = 0
  var _k: Int = 5
  var _weightCf: Double = 0.8
  var _weightSequential: Double = 0.2


  def setNumberOfItems(numberOfItems: Int): this.type = {
    this.numberOfItems = numberOfItems
    this
  }

  def setCF(recsys: ExplicitBaseRecommender): this.type = {
    this.collaborativeFiltering = recsys
    this
  }

  def setSequential(recsys: SequentialTopKRecommender): this.type = {
    this.sequential = recsys
    this
  }

  def fit(train: DataFrame): Unit = {
    this.collaborativeFiltering.fit(
      train, this.numberOfItems
    )

    this.sequential.fit(train)
  }

  def transform(test: DataFrame): Set[Int] = {
    val predictionsSequential = this.sequential.transform(
      test
    ).map(element => {
      (element._1, element._2 * this._weightSequential)
    })

    val explicitArray = Vectors.sparse(
      this.numberOfItems,
      test.select("item_id", "rating").collect().map(row => {
        (row.getInt(0) - 1, 1.0)
      })
    ).toDense.toArray

    val predictionsCF = this.collaborativeFiltering.transform(
      explicitArray
    ).map(element => {
      (element._1, element._2 * this._weightCf)
    })

    val combination = (predictionsSequential ++ predictionsCF).groupBy(_._1).mapValues(
      _.map(_._2).sum
    ).toArray

    combination.sortWith(_._2 > _._2).take(this._k).map(_._1).toSet
  }
}
