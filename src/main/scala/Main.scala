import scala.math.{pow, sqrt}

import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list}

import accumulator.ListBufferAccumulator
import recommender.user_based.UserBasedRatingRecommender
import recommender.item_based.ItemBasedRatingRecommender
import similarity._

object Main {
  def dataset(session: SparkSession, filename: String): DataFrame = {
    session.read.options(
      Map("header" -> "false", "delimiter" -> "\t")
    ).schema(
      StructType(
        Seq(
          StructField("user_id", IntegerType, nullable = false),
          StructField("item_id", IntegerType, nullable = false),
          StructField("rating", DoubleType, nullable = false),
          StructField("timestamp", LongType, nullable = false)
        )
      )
    ).csv(filename).drop("timestamp")
  }

  def userBasedCrossValidation(spark: SparkSession, similarity: BaseSimilarity, k: Int): Seq[Double] = {
    val recSysUserBased = new UserBasedRatingRecommender(k)
    recSysUserBased.setSimilarityMeasure(similarity)

    val predictions_accumulator1 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator1, "predictions1")
    val predictions_accumulator2 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator2, "predictions2")
    val predictions_accumulator3 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator3, "predictions3")
    val predictions_accumulator4 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator4, "predictions4")
    val predictions_accumulator5 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator5, "predictions5")

    Seq(1, 2, 3, 4, 5).map(index => {
      println("Fold " + index)
      val train = dataset(spark, "train-fold" + index + ".csv")
      val test = dataset(spark, "test-fold" + index + ".csv")

      val accumulator = index match {
        case 1 => predictions_accumulator1
        case 2 => predictions_accumulator2
        case 3 => predictions_accumulator3
        case 4 => predictions_accumulator4
        case 5 => predictions_accumulator5
      }

      recSysUserBased.readDataframe(spark, train, 1682)

      test.groupBy("user_id").agg(
        collect_list(col("item_id")).as("items"),
        collect_list(col("rating")).as("ratings")
      ).foreach(row => {
        val userId = row.getInt(0)
        val items = row.getList(1).toArray()
        val ratings = row.getList(2).toArray()

        val item = recSysUserBased._matrix.rowIter.slice(userId - 1, userId).toList.head.toArray
        items.zip(ratings).foreach(a => {
          val prediction = recSysUserBased.predictionRatingItem(
            item, a._1.asInstanceOf[Int]
          )

          val difference = prediction - a._2.asInstanceOf[Double]
          accumulator.add(difference)
        }: Unit)
      }: Unit)

      sqrt(accumulator.value.map(pow(_, 2)).sum / accumulator.value.length)
    })
  }

  def itemBasedCrossValidation(spark: SparkSession, similarity: BaseSimilarity, k: Int): Seq[Double] = {
    val recSysItemBased = new ItemBasedRatingRecommender(k)
    recSysItemBased.setSimilarityMeasure(similarity)

    val predictions_accumulator1 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator1, "predictions1")
    val predictions_accumulator2 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator2, "predictions2")
    val predictions_accumulator3 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator3, "predictions3")
    val predictions_accumulator4 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator4, "predictions4")
    val predictions_accumulator5 = new ListBufferAccumulator[Double]
    spark.sparkContext.register(predictions_accumulator5, "predictions5")

    Seq(1, 2, 3, 4, 5).map(index => {
      println("Fold " + index)
      val train = dataset(spark, "train-fold" + index + ".csv")
      val test = dataset(spark, "test-fold" + index + ".csv")

      val accumulator = index match {
        case 1 => predictions_accumulator1
        case 2 => predictions_accumulator2
        case 3 => predictions_accumulator3
        case 4 => predictions_accumulator4
        case 5 => predictions_accumulator5
      }

      recSysItemBased.readDataframe(spark, train, 1682)

      test.groupBy("item_id").agg(
        collect_list(col("user_id")).as("users"),
        collect_list(col("rating")).as("ratings")
      ).foreach(row => {
        val itemId = row.getInt(0)
        val users = row.getList(1).toArray()
        val ratings = row.getList(2).toArray()

        val item = recSysItemBased._matrix.rowIter.slice(itemId - 1, itemId).toList.head.toArray
        users.zip(ratings).foreach(a => {
          val prediction = recSysItemBased.predictionRatingItem(
            item, a._1.asInstanceOf[Int]
          )

          val difference = prediction - a._2.asInstanceOf[Double]
          accumulator.add(difference)
        }: Unit)
      }: Unit)

      sqrt(accumulator.value.map(pow(_, 2)).sum / accumulator.value.length)
    })
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("TFM").getOrCreate()

    val crossValidationResults = userBasedCrossValidation(spark, new CosineSimilarity, 25)

    crossValidationResults.zipWithIndex.foreach(f => {
      println("Fold " + (f._2 + 1) + ": " + f._1)
    })

    spark.stop()
  }
}
