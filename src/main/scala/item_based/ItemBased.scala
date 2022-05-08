package item_based

import accumulators.{DoubleListBufferAccumulator, IntListBufferAccumulator}
import org.apache.spark.ml.linalg.{DenseMatrix, SparseMatrix, Vector}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import similarity.Similarity

import scala.math.abs

class ItemBased(session: SparkSession) {
  var spark: SparkSession = session
  var dataframe: DataFrame = null
  var itemUserMatrix: DenseMatrix = null
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
    ).transpose.toDense

    this.itemUserMatrix = denseMatrix
  }

  def getKSimilarItems(targetItem: Array[Double], k: Int): List[(Double, Vector)] = {
    val correlations = this.itemUserMatrix.rowIter.map(
      f => this.similarity.getSimilarity(targetItem, f.toArray)
    ).toList

    correlations.zip(this.itemUserMatrix.rowIter).sortWith(_._1 > _._1).take(k)
  }

/*  def predictionRatingItem(targetItem: Array[Double], item: Int): Double = {
    val topKItems = this.getKSimilarItems(targetItem, 25)
    val ratingMean = targetItem.sum / targetItem.length

    val numerator = topKItems.map(a => {
      a._1 * a._2(item)
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    ratingMean + (numerator/denominator)
  }

  def topKItemsForUser(targetItem: Array[Double], k: Int): List[(Int, Double)] = {
    val unratedItems = targetItem.zipWithIndex.filter(_._1 == 0).map(_._2)

    unratedItems.map(item => {
      (item, this.predictionRatingItem(targetItem, item))
    }).sortWith(_._1 > _._1).take(k).toList
  }*/
}