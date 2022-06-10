package recommender.content

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseMatrix, SparseMatrix}
import org.apache.spark.sql.functions.{col, collect_list}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import accumulator.ListBufferAccumulator
import similarity.BaseSimilarity


class BaseRecommender extends Serializable {
  var _features: Array[Row] = null
  var _matrix: DenseMatrix = null
  var _similarity: BaseSimilarity = null

  def setSimilarityMeasure(similarityMeasure: BaseSimilarity): Unit = {
    this._similarity = similarityMeasure
  }

  def setFeatures(features: DataFrame): Unit = {
    this._features = this.transformFeatures(features)
  }

  def readDataframe(session: SparkSession, dataframe: DataFrame, numberOfItems: Long): Unit = {
    val numberOfUsers = this.getNumberOfUsers(dataframe)
    this._matrix = this.calculateDenseMatrix(session, dataframe, numberOfUsers, numberOfItems)
  }

  protected def transformFeatures(features: DataFrame): Array[Row] = {
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

    transformed.collect()
  }

  protected def getNumberOfUsers(dataframe: DataFrame): Long = {
    dataframe.select("user_id").distinct().count()
  }

  protected def getNotRepresentedItems(groupedDf: DataFrame, cols: Long): Seq[Int] = {
    val everyItem = Range.inclusive(1, cols.toInt).toSet
    val actualItems = groupedDf.select(
      "item_id"
    ).collect().map(_.getInt(0)).toSet

    everyItem.diff(actualItems).toSeq.sorted
  }

  protected def createAndRegisterAccumulators(session: SparkSession): (ListBufferAccumulator[Long], ListBufferAccumulator[Long], ListBufferAccumulator[Double]) = {
    val rowIndices = new ListBufferAccumulator[Long]
    val colSeparators = new ListBufferAccumulator[Long]
    val values = new ListBufferAccumulator[Double]

    session.sparkContext.register(rowIndices, "ratings")
    session.sparkContext.register(colSeparators, "col_separator")
    session.sparkContext.register(values, "row_indices")

    (rowIndices, colSeparators, values)
  }

  protected def calculateDenseMatrix(session: SparkSession, dataframe: DataFrame, rows: Long, cols: Long): DenseMatrix = {
    val groupedDf = dataframe.groupBy(
      "item_id"
    ).agg(
      collect_list(col("user_id")).as("users"),
      collect_list(col("rating")).as("ratings")
    ).drop("user_id", "rating")

    val notRepresentedItems = this.getNotRepresentedItems(groupedDf, cols)
    val (rowIndices, colSeparators, values) = this.createAndRegisterAccumulators(session)

    groupedDf.foreach((row: Row) => {
      val users = row.getList(1).toArray()
      val ratings = row.getList(2).toArray()

      users.zip(ratings).foreach(UserRatingTuple => {
        rowIndices.add(UserRatingTuple._1.asInstanceOf[Int] - 1)
        values.add(UserRatingTuple._2.asInstanceOf[Double])
      })

      colSeparators.add(values.value.length)
    })

    val separators: ListBuffer[Long] = 0.toLong +: colSeparators.value

    notRepresentedItems.foreach(index => {
      separators.insert(
        index - 1,
        separators(index - 1)
      )
    })

    val sparse = new SparseMatrix(
      numRows = rows.toInt,
      numCols = cols.toInt,
      colPtrs = separators.toArray.map(_.toInt),
      rowIndices = rowIndices.value.toArray.map(_.toInt),
      values = values.value.toArray
    )

    sparse.transpose.toDense
  }
}
