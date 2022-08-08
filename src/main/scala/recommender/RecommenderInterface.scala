package recommender

import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.sql.DataFrame

import similarity.BaseSimilarity
import accumulator.ListBufferAccumulator


abstract class RecommenderInterface(numberOfItems: Long, isUserBased: Boolean = true) extends Serializable {
  protected var _isUserBased: Boolean = isUserBased
  var _matrix: DenseMatrix = null
  protected var _similarity: BaseSimilarity = null
  protected var _numberOfItems: Long = numberOfItems

  def getSimilarity: BaseSimilarity
  def setSimilarity(similarity: BaseSimilarity): Unit

  def getIsUserBased: Boolean
  def setIsUserBased(isUserBased: Boolean): Unit

  def getNumberOfItems: Long
  def setNumberOfItems(numberOfItems: Long): Unit

  def fit(dataframe: DataFrame): Unit
  def transform(target: Array[Double], index: Int): Double
  def transform(target: Array[Double]): Seq[(Int, Double)]
  def transform(target: DataFrame): Seq[(Int, Double)]

  protected def getNumberOfUsers(dataframe: DataFrame): Long
  protected def getNotRepresentedItems(dataframeGroupByItem: DataFrame): Seq[Int]
  protected def createAndRegisterAccumulators: (ListBufferAccumulator[Long], ListBufferAccumulator[Long], ListBufferAccumulator[Double])
  protected def solveMatrix(dataframe: DataFrame, numberOfUsers: Long): DenseMatrix
}
