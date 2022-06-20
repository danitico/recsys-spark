import scala.math.{pow, sqrt}
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list, from_unixtime, udf}
import accumulator.ListBufferAccumulator
import metrics.TopKMetrics
import org.apache.spark.ml.linalg.SparseVector
import recommender.collaborative.user_based.{UserBasedRatingRecommender, UserBasedTopKRecommender}
import recommender.collaborative.item_based.ItemBasedRatingRecommender
import recommender.content.ContentBasedRatingRecommender
import recommender.sequential.TopKSequentialRecommender
import similarity._

object Main {
  def dataset(filename: String): DataFrame = {
    val session = SparkSession.getActiveSession.orNull

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
    ).csv(filename).withColumn("timestamp", from_unixtime(col("timestamp")))
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
      val train = dataset("data/train-fold" + index + ".csv")
      val test = dataset("data/test-fold" + index + ".csv")

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
      val train = dataset("data/train-fold" + index + ".csv")
      val test = dataset("data/test-fold" + index + ".csv")

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

  def userBasedTopKCrossValidation(spark: SparkSession, similarity: BaseSimilarity, kUsers: Int, topK: Int): Seq[Double] = {
    val recSys = new UserBasedTopKRecommender(kUsers, topK)
    recSys.setSimilarityMeasure(similarity)

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
      val train = dataset("data/train-fold" + index + ".csv")
      val test = dataset("data/test-fold" + index + ".csv")

      val accumulator = index match {
        case 1 => predictions_accumulator1
        case 2 => predictions_accumulator2
        case 3 => predictions_accumulator3
        case 4 => predictions_accumulator4
        case 5 => predictions_accumulator5
      }

      recSys.readDataframe(spark, train, 1682)

      test.groupBy("user_id").agg(
        collect_list(col("item_id")).as("items"),
        collect_list(col("rating")).as("ratings")
      ).foreach(row => {
        val userId = row.getInt(0)
        val items = row.getList(1).toArray()
        val ratings = row.getList(2).toArray()

        val relevant = items.zip(ratings).filter(
          _._2.asInstanceOf[Double] >= 4
        ).map(_._1.asInstanceOf[Int]).toSet

        val user = recSys._matrix.rowIter.slice(userId - 1, userId).toList.head.toArray
        val selected = recSys.topKItemsForUser(user)

        accumulator.add(
          new TopKMetrics(k = 5, selected, relevant).getPrecision
        )
      }: Unit)

      accumulator.value.sum / accumulator.value.length
    })
  }

  def contentCrossValidation(spark: SparkSession, similarity: BaseSimilarity, k: Int): Seq[Double] = {
    val recSys = new ContentBasedRatingRecommender(k)
    recSys.setSimilarityMeasure(similarity)

    val features = spark.read.options(
      Map("header" -> "true", "inferSchema" -> "true")
    ).csv("data/features.csv").cache()
    recSys.setFeatures(features)

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
      val train = dataset("data/train-fold" + index + ".csv")
      val test = dataset("data/test-fold" + index + ".csv")

      val accumulator = index match {
        case 1 => predictions_accumulator1
        case 2 => predictions_accumulator2
        case 3 => predictions_accumulator3
        case 4 => predictions_accumulator4
        case 5 => predictions_accumulator5
      }

      recSys.readDataframe(spark, train, 1682)

      test.groupBy("item_id").agg(
        collect_list(col("user_id")).as("users"),
        collect_list(col("rating")).as("ratings")
      ).foreach(row => {
        val itemId = row.getInt(0)
        val users = row.getList(1).toArray()
        val ratings = row.getList(2).toArray()

        val itemFeature = recSys._features.filter(
          _.getInt(0) == itemId
        ).head.getAs[SparseVector](1)
        users.zip(ratings).foreach(a => {
          val prediction = recSys.predictionRatingItem(
            itemFeature.toDense.toArray, a._1.asInstanceOf[Int]
          )

          val difference = prediction - a._2.asInstanceOf[Double]
          accumulator.add(difference)
        }: Unit)
      }: Unit)

      sqrt(accumulator.value.map(pow(_, 2)).sum / accumulator.value.length)
    })
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master(
      "local[*]"
    ).config(
      "spark.sql.autoBroadcastJoinThreshold", "-1"
    ).config(
      "spark.jars", "/home/daniel/Desktop/recommendations/lib/sparkml-som_2.12-0.2.1.jar"
    ).appName(
      "TFM"
    ).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val recsys = new TopKSequentialRecommender().setKCustomer(
      5
    ).setNumberItems(1682).setKMeansDistance("cosine").setPeriods(3)

    val train = dataset("data/train-fold1.csv")

    recsys.fit(train)

    spark.stop()
  }
}
