package user_based

import scala.math.abs

import org.apache.spark.ml.linalg.{DenseMatrix, SparseMatrix, Vector}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

import accumulators.{DoubleListBufferAccumulator, IntListBufferAccumulator}
import similarity.Similarity


class UserBased(session: SparkSession) {
  var spark: SparkSession = session
  var dataframe: DataFrame = null
  var userItemMatrix: DenseMatrix = null
  var similarity: Similarity = null

  def readDataset(filename: String): Unit = {
    this.dataframe = this.spark.read.options(
      Map("header" -> "true")
    ).schema(
      StructType(
        Seq(
          StructField("user_id", IntegerType, nullable = false),
          StructField("item_id", IntegerType, nullable = false),
          StructField("rating", DoubleType, nullable = false),
          StructField("timestamp", IntegerType, nullable = false)
        )
      )
    ).csv(
      filename,
    ).drop(
      "timestamp"
    )
  }

  def setSimilarityMeasure(similarityMeasure: Similarity): Unit = {
    this.similarity = similarityMeasure
  }

  def getNumberUserAndItems: (Long, Long) = {
    val numberUsers = this.dataframe.select("user_id").distinct().count()
    val numberItems = this.dataframe.select("item_id").distinct().count()

    (numberUsers, numberItems)
  }

  def createAndRegisterAccumulators: (IntListBufferAccumulator, IntListBufferAccumulator, DoubleListBufferAccumulator) = {
    val rowIndices = new IntListBufferAccumulator
    val colSeparators = new IntListBufferAccumulator
    val values = new DoubleListBufferAccumulator

    this.spark.sparkContext.register(rowIndices, "ratings")
    this.spark.sparkContext.register(colSeparators, "col_separator")
    this.spark.sparkContext.register(values, "row_indices")

    (rowIndices, colSeparators, values)
  }

  def calculateDenseMatrix(): Unit = {
    val (numberUsers, numberItems) = this.getNumberUserAndItems
    val groupedDf = this.dataframe.groupBy("item_id").agg(
      collect_list(col("user_id")).as("users"),
      collect_list(col("rating")).as("ratings")
    ).drop("user_id").drop("rating")

    val (rowIndices, colSeparators, values) = this.createAndRegisterAccumulators

    groupedDf.foreach((row: Row) => {
      val users = row.getList(1).toArray()
      val ratings = row.getList(2).toArray()

      users.zip(ratings).foreach(UserRatingTuple => {
        rowIndices.add(UserRatingTuple._1.asInstanceOf[Int] - 1)
        values.add(UserRatingTuple._2.asInstanceOf[Double])
      })

      colSeparators.add(values.value.length)
    }: Unit)

    val denseMatrix = new SparseMatrix(
      numRows = numberUsers.toInt,
      numCols = numberItems.toInt,
      colPtrs = 0 +: colSeparators.value.toArray,
      rowIndices = rowIndices.value.toArray,
      values = values.value.toArray
    ).toDense

    this.userItemMatrix = denseMatrix
  }

  def getKSimilarUsers(targetUser: Array[Double], k: Int): List[(Double, Vector)] = {
    val correlations = this.userItemMatrix.rowIter.map(
      f => this.similarity.getSimilarity(targetUser, f.toArray)
    ).toList

    correlations.zip(this.userItemMatrix.rowIter).sortWith(_._1 > _._1).take(k)
  }

  def ratingCalculation(topKUsers: List[(Double, Vector)], ratingMean: Double, item: Int): Double = {
    val numerator = topKUsers.map(a => {
      a._1 * a._2(item)
    }).sum

    val denominator = topKUsers.map(_._1).reduce(abs(_) + abs(_))

    ratingMean + (numerator/denominator)
  }

  def predictionRatingItem(targetUser: Array[Double], item: Int): Double = {
    val topKUsers = this.getKSimilarUsers(targetUser, 25)
    val ratingMean = targetUser.sum / targetUser.length

    this.ratingCalculation(topKUsers, ratingMean, item)
  }

  def topKItemsForUser(targetUser: Array[Double], k: Int): List[(Int, Double)] = {
    val unratedItems = targetUser.zipWithIndex.filter(_._1 == 0).map(_._2)
    val ratingMean = targetUser.sum / targetUser.length
    val topKUsers = this.getKSimilarUsers(targetUser, 25)

    unratedItems.map(item => {
      (item, this.ratingCalculation(topKUsers, ratingMean, item))
    }).sortWith(_._2 > _._2).take(k).toList
  }
}
