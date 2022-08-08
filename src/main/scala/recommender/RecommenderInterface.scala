package recommender

import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.sql.DataFrame

import similarity.BaseSimilarity
import accumulator.ListBufferAccumulator


abstract class RecommenderInterface(numberOfItems: Long, isUserBased: Boolean = true) extends Serializable {
  var _isUserBased: Boolean = isUserBased
  var _matrix: DenseMatrix = null
  var _similarity: BaseSimilarity = null
  var _numberOfItems: Long = numberOfItems

  def getSimilarity: BaseSimilarity
  def setSimilarity(similarity: BaseSimilarity): Unit

  def getIsUserBased: Boolean
  def setIsUserBased(isUserBased: Boolean): Unit

  def getNumberOfItems: Long
  def setNumberOfItems(numberOfItems: Long): Unit

  def fit(dataframe: DataFrame): Unit
  def transform(target: Array[Double], index: Int): Double
  def transform(target: Array[Double]): Seq[(Int, Double)]

  def getNumberOfUsers(dataframe: DataFrame): Long
  def getNotRepresentedItems(dataframeGroupByItem: DataFrame): Seq[Int]
  def createAndRegisterAccumulators: (ListBufferAccumulator[Long], ListBufferAccumulator[Long], ListBufferAccumulator[Double])
  def solveMatrix(dataframe: DataFrame, numberOfUsers: Long): DenseMatrix
}
