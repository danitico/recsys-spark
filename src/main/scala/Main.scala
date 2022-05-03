import scala.collection.mutable.ListBuffer
import scala.math.{pow, sqrt}

import org.apache.spark.ml.linalg.{DenseMatrix, SparseMatrix, Vectors}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.util.AccumulatorV2


class IntListBufferAccumulator extends AccumulatorV2[Int, ListBuffer[Int]] {
  private val accumulator = ListBuffer[Int]()

  def add(element: Int): Unit = {
    this.accumulator += element
  }

  def copy(): IntListBufferAccumulator = {
    this
  }

  def isZero: Boolean = {
    this.accumulator.isEmpty
  }

  def merge(other: AccumulatorV2[Int, ListBuffer[Int]]): Unit = {
    this.accumulator.addAll(other.value)
  }

  def reset(): Unit = {
    this.accumulator.clear()
  }

  def value: ListBuffer[Int] = {
    this.accumulator
  }
}

class DoubleListBufferAccumulator extends AccumulatorV2[Double, ListBuffer[Double]] {
  private val accumulator = ListBuffer[Double]()

  def add(element: Double): Unit = {
    this.accumulator += element
  }

  def copy(): DoubleListBufferAccumulator = {
    this
  }

  def isZero: Boolean = {
    this.accumulator.isEmpty
  }

  def merge(other: AccumulatorV2[Double, ListBuffer[Double]]): Unit = {
    this.accumulator.addAll(other.value)
  }

  def reset(): Unit = {
    this.accumulator.clear()
  }

  def value: ListBuffer[Double] = {
    this.accumulator
  }
}


object Main {
  def readDataset(spark: SparkSession, filename: String): DataFrame = {
    spark.read.options(
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

  def getNumberUserAndItems(df: DataFrame): (Long, Long) = {
    val numberUsers = df.select("user_id").distinct().count()
    val numberItems = df.select("item_id").distinct().count()

    (numberUsers, numberItems)
  }

  def createAndRegisterAccumulators(spark: SparkSession): (IntListBufferAccumulator, IntListBufferAccumulator, DoubleListBufferAccumulator) = {
    val rowIndices = new IntListBufferAccumulator
    val colSeparators = new IntListBufferAccumulator
    val values = new DoubleListBufferAccumulator

    spark.sparkContext.register(rowIndices, "ratings")
    spark.sparkContext.register(colSeparators, "col_separator")
    spark.sparkContext.register(values, "row_indices")

    (rowIndices, colSeparators, values)
  }

  def toDenseMatrix(spark: SparkSession, df: DataFrame): DenseMatrix = {
    val (numberUsers, numberItems) = this.getNumberUserAndItems(df)
    val groupedDf = df.groupBy("item_id").agg(
      collect_list(col("user_id")).as("users"),
      collect_list(col("rating")).as("ratings")
    ).drop("user_id").drop("rating")

    val (rowIndices, colSeparators, values) = this.createAndRegisterAccumulators(spark)

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

    denseMatrix
  }

  def pearsonSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double = {
    val mean1 = firstArray.sum / firstArray.length
    val mean2 = secondArray.sum / secondArray.length

    val differencesFirstArray = firstArray.map(_ - mean1)
    val differencesSecondArray = secondArray.map(_ - mean2)
    val differencesFirstArraySquared = differencesFirstArray.map(pow(_, 2))
    val differencesSecondArraySquared = differencesSecondArray.map(pow(_, 2))

    val numerator = differencesFirstArray.zip(differencesSecondArray).map {case (a, b) => a * b}.sum
    val denominator = sqrt(differencesFirstArraySquared.sum) * sqrt(differencesSecondArraySquared.sum)

    numerator/denominator
  }

  def cosineSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double = {
    val numerator = firstArray.zip(secondArray).map {case (a, b) => a*b}.sum
    val denominator = sqrt(
      firstArray.map(pow(_, 2)).sum
    ) * sqrt(
      secondArray.map(pow(_, 2)).sum
    )

    numerator / denominator
  }

  def euclideanSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double = {
    // sum one to the denominator in order to avoid division by zero

    1 / (sqrt(firstArray.zip(secondArray).map {case (a, b) => pow(a - b, 2)}.sum) + 1)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("TFM").getOrCreate()
    val df = this.readDataset(spark, "data.csv")
    val userItemMatrix = this.toDenseMatrix(spark, df)

    val newUser = Vectors.sparse(1682, Seq((0, 1.0), (3, 1.0))).toDense.toArray
    val indices = newUser.zipWithIndex.filter(_._1 > 0).map(_._2)

    val correlations = userItemMatrix.rowIter.map(
      f => this.pearsonSimilarity(f.toArray, newUser)
    ).toList

    val topKUsers = correlations.zip(userItemMatrix.rowIter).sortWith(_._1 > _._1).take(5).map(_._2.toArray)

    val frequenciesItems = topKUsers.reduce((a, b) => {
      a.zip(b).map {case (c, d) => (if (c > 0) 1 else 0) + (if (d > 0) 1 else 0)}
    })

    indices.foreach(
      frequenciesItems(_) = 0.0
    )

    val topNItems = frequenciesItems.zipWithIndex.filter(_._1 > 0).sortWith(_._1 > _._1).take(5).map(_._2)

    topNItems.map(_ + 1).foreach(println(_))
    spark.stop()
  }
}
