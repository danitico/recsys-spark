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
  var totalNumberOfItems: Int = -1

  def readDataset(filename: String, totalNumberOfItems: Int): Unit = {
    this.totalNumberOfItems = totalNumberOfItems
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

  def getNumberUserAndItems: (Long, Int) = {
    val numberUsers = this.dataframe.select("user_id").distinct().count()

    (numberUsers, this.totalNumberOfItems)
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

    val everyItem = Set.range(1, numberItems)
    val actualItems = groupedDf.select(
      "item_id"
    ).collect().map(_.getInt(0)).toSet
    val notRatedItems = everyItem -- actualItems

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

    val separators = 0 +: colSeparators.value
    notRatedItems.foreach(index => {
      separators.insert(
        index - 1,
        separators(index - 1)
      )
    })

    val denseMatrix = new SparseMatrix(
      numRows = numberUsers.toInt,
      numCols = numberItems,
      colPtrs = separators.toArray,
      rowIndices = rowIndices.value.toArray,
      values = values.value.toArray
    ).transpose.toDense

    this.itemUserMatrix = denseMatrix
  }

  def getKSimilarItems(targetItem: Array[Double], k: Int, user: Int): List[(Double, Vector)] = {
    val itemsWithRating = this.itemUserMatrix.rowIter.filter(_(user) > 0).toList

    val correlations = itemsWithRating.map(
      f => this.similarity.getSimilarity(targetItem, f.toArray)
    )

    correlations.zip(itemsWithRating).sortWith(_._1 > _._1).take(k)
  }

  def ratingCalculation(topKItems: List[(Double, Vector)], ratingMean: Double, user: Int): Double = {
    val numerator = topKItems.map(a => {
      a._1 * a._2(user)
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    ratingMean + (numerator/denominator)
  }

  def predictionRatingItem(targetItem: Array[Double], user: Int): Double = {
    val topKItems = this.getKSimilarItems(targetItem, 25, user)
    val ratingMean = targetItem.sum / targetItem.length

    this.ratingCalculation(topKItems, ratingMean, user)
  }
}
