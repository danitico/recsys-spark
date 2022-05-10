package recommenders

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.linalg.{DenseMatrix, SparseMatrix}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list}

import accumulators.{DoubleListBufferAccumulator, LongListBufferAccumulator}
import similarity.Similarity


class BaseRecommender(session: SparkSession) {
  var spark: SparkSession = session
  var matrix: DenseMatrix = null
  var similarity: Similarity = null

  def setSimilarityMeasure(similarityMeasure: Similarity): Unit = {
    this.similarity = similarityMeasure
  }

  def readDataframe(dataframe: DataFrame, numberOfItems: Long): Unit = {
    val numberOfUsers = this.getNumberOfUsers(dataframe)
    this.matrix = this.calculateDenseMatrix(dataframe, numberOfUsers, numberOfItems)
  }

  protected def getNumberOfUsers(dataframe: DataFrame): Long = {
    dataframe.select("user_id").distinct().count()
  }

  protected def getNotRepresentedItems(groupedDf: DataFrame, cols: Long): Set[Long] = {
    val everyItem = Set.range(1, cols)
    val actualItems = groupedDf.select(
      "item_id"
    ).collect().map(_.getInt(0).toLong).toSet

    everyItem -- actualItems
  }

  protected def createAndRegisterAccumulators: (LongListBufferAccumulator, LongListBufferAccumulator, DoubleListBufferAccumulator) = {
    val rowIndices = new LongListBufferAccumulator
    val colSeparators = new LongListBufferAccumulator
    val values = new DoubleListBufferAccumulator

    this.spark.sparkContext.register(rowIndices, "ratings")
    this.spark.sparkContext.register(colSeparators, "col_separator")
    this.spark.sparkContext.register(values, "row_indices")

    (rowIndices, colSeparators, values)
  }

  protected def calculateDenseMatrix(dataframe: DataFrame, rows: Long, cols: Long): DenseMatrix = {
    val groupedDf = dataframe.groupBy(
      "item_id"
    ).agg(
      collect_list(col("user_id")).as("users"),
      collect_list(col("rating")).as("ratings")
    ).drop("user_id", "rating")

    val notRepresentedItems = this.getNotRepresentedItems(groupedDf, cols)
    val (rowIndices, colSeparators, values) = this.createAndRegisterAccumulators

    groupedDf.foreach((row: Row) => {
      val users = row.getList(1).toArray()
      val ratings = row.getList(2).toArray()

      users.zip(ratings).foreach(UserRatingTuple => {
        rowIndices.add(UserRatingTuple._1.asInstanceOf[Int] - 1)
        values.add(UserRatingTuple._2.asInstanceOf[Double])
      })

      colSeparators.add(values.value.length)
    })

    val separators: ListBuffer[Long] = 0 +: colSeparators.value

    notRepresentedItems.foreach(index => {
      separators.insert(
        index.toInt - 1,
        separators(index.toInt - 1)
      )
    })

    val sparse = new SparseMatrix(
      numRows = rows.toInt,
      numCols = cols.toInt,
      colPtrs = separators.toArray.map(_.toInt),
      rowIndices = rowIndices.value.toArray.map(_.toInt),
      values = values.value.toArray
    )

    sparse.toDense
  }
}
