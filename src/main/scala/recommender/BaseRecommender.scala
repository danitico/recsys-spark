package recommender

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.linalg.{DenseMatrix, SparseMatrix}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list}

import accumulator.ListBufferAccumulator
import similarity.BaseSimilarity


class BaseRecommender(session: SparkSession, isUserBased: Boolean = true) {
  var _spark: SparkSession = session
  var _isUserBased: Boolean = isUserBased
  var _matrix: DenseMatrix = null
  var _similarity: BaseSimilarity = null

  def setSimilarityMeasure(similarityMeasure: BaseSimilarity): Unit = {
    this._similarity = similarityMeasure
  }

  def readDataframe(dataframe: DataFrame, numberOfItems: Long): Unit = {
    val numberOfUsers = this.getNumberOfUsers(dataframe)
    this._matrix = this.calculateDenseMatrix(dataframe, numberOfUsers, numberOfItems)
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

  protected def createAndRegisterAccumulators: (ListBufferAccumulator[Long], ListBufferAccumulator[Long], ListBufferAccumulator[Double]) = {
    val rowIndices = new ListBufferAccumulator[Long]
    val colSeparators = new ListBufferAccumulator[Long]
    val values = new ListBufferAccumulator[Double]

    this._spark.sparkContext.register(rowIndices, "ratings")
    this._spark.sparkContext.register(colSeparators, "col_separator")
    this._spark.sparkContext.register(values, "row_indices")

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

    if (this._isUserBased) {
      sparse.toDense
    } else {
      sparse.transpose.toDense
    }
  }
}
