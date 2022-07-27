import scala.math.{pow, sqrt}
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list, from_unixtime}
import accumulator.ListBufferAccumulator
import metrics.{PredictionMetrics, RankingMetrics}
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import recommender.collaborative.`implicit`.user_based.ImplicitUserBasedTopKRecommender
import recommender.collaborative.explicit.user_based.{UserBasedRatingRecommender, UserBasedTopKRecommender}
import recommender.collaborative.explicit.item_based.ItemBasedRatingRecommender
import recommender.content.ContentBasedRatingRecommender
import recommender.sequential.SequentialTopKRecommender
import recommender.hybrid.HybridRecommenderTopK
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
    ).csv(filename).withColumn(
      "timestamp",
      from_unixtime(col("timestamp"))
    )
  }

  def userBasedCrossValidation(spark: SparkSession, similarity: BaseSimilarity, k: Int): Seq[(Double, Double)] = {
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

      recSysUserBased.fit(train, 1682)

      test.groupBy("user_id").agg(
        collect_list(col("item_id")).as("items"),
        collect_list(col("rating")).as("ratings")
      ).foreach(row => {
        val userId = row.getInt(0)
        val items = row.getList(1).toArray()
        val ratings = row.getList(2).toArray()

        val item = recSysUserBased._matrixRows.slice(userId - 1, userId).head.toArray
        items.zip(ratings).foreach(a => {
          val prediction = recSysUserBased.transform(
            item, a._1.asInstanceOf[Int]
          )

          val difference = prediction - a._2.asInstanceOf[Double]
          accumulator.add(difference)
        }: Unit)
      }: Unit)

      val metrics = new PredictionMetrics(accumulator.value.toArray)

      metrics.getPredictionMetrics
    })
  }

  def itemBasedCrossValidation(spark: SparkSession, similarity: BaseSimilarity, k: Int): Seq[(Double, Double)] = {
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

      recSysItemBased.fit(train, 1682)

      test.groupBy("item_id").agg(
        collect_list(col("user_id")).as("users"),
        collect_list(col("rating")).as("ratings")
      ).foreach(row => {
        val itemId = row.getInt(0)
        val users = row.getList(1).toArray()
        val ratings = row.getList(2).toArray()

        val item = recSysItemBased._matrixRows.slice(itemId - 1, itemId).toList.head.toArray
        users.zip(ratings).foreach(a => {
          val prediction = recSysItemBased.transform(
            item, a._1.asInstanceOf[Int]
          )

          val difference = prediction - a._2.asInstanceOf[Double]
          accumulator.add(difference)
        }: Unit)
      }: Unit)

      val metrics = new PredictionMetrics(accumulator.value.toArray)

      metrics.getPredictionMetrics
    })
  }

  def userBasedTopKCrossValidation(spark: SparkSession, similarity: BaseSimilarity, kUsers: Int, topK: Int): Seq[(Double, Double, Double)] = {
    val recSys = new UserBasedTopKRecommender(kUsers, topK)
    recSys.setSimilarityMeasure(similarity)

    val predictions_accumulator1 = new ListBufferAccumulator[(Double, Double, Double)]
    spark.sparkContext.register(predictions_accumulator1, "predictions1")
    val predictions_accumulator2 = new ListBufferAccumulator[(Double, Double, Double)]
    spark.sparkContext.register(predictions_accumulator2, "predictions2")
    val predictions_accumulator3 = new ListBufferAccumulator[(Double, Double, Double)]
    spark.sparkContext.register(predictions_accumulator3, "predictions3")
    val predictions_accumulator4 = new ListBufferAccumulator[(Double, Double, Double)]
    spark.sparkContext.register(predictions_accumulator4, "predictions4")
    val predictions_accumulator5 = new ListBufferAccumulator[(Double, Double, Double)]
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

      recSys.fit(train, 1682)

      test.groupBy("user_id").agg(
        collect_list(col("item_id")).as("items"),
        collect_list(col("rating")).as("ratings")
      ).foreach(row => {
        val userId = row.getInt(0)
        val items = row.getList(1).toArray()
        val ratings = row.getList(2).toArray()

        val relevant = items.zip(ratings).filter(
          _._2.asInstanceOf[Double] >= 4.0
        ).map(_._1.asInstanceOf[Int]).toSet

        val user = recSys._matrixRows.slice(userId - 1, userId).toList.head.toArray
        val selected = recSys.transform(user)

        accumulator.add(
          new RankingMetrics(k = topK, selected, relevant).getRankingMetrics
        )
      }: Unit)

      val metricPerUser = accumulator.value
      val sumMetrics = metricPerUser.reduce((a, b) => {
        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
      })

      val finalMetrics = (
        sumMetrics._1 / metricPerUser.length,
        sumMetrics._2 / metricPerUser.length,
        sumMetrics._3 / metricPerUser.length
      )
      println(finalMetrics)
      finalMetrics
    })
  }

  def itemBasedTopKCrossValidation(spark: SparkSession, similarity: BaseSimilarity, kItems: Int, topK: Int): Seq[(Double, Double, Double)] = {
    val recSys = new ItemBasedTopKRecommender(kItems, topK)
    recSys.setSimilarityMeasure(similarity)

    val predictions_accumulator1 = new ListBufferAccumulator[(Double, Double, Double)]
    spark.sparkContext.register(predictions_accumulator1, "predictions1")
    val predictions_accumulator2 = new ListBufferAccumulator[(Double, Double, Double)]
    spark.sparkContext.register(predictions_accumulator2, "predictions2")
    val predictions_accumulator3 = new ListBufferAccumulator[(Double, Double, Double)]
    spark.sparkContext.register(predictions_accumulator3, "predictions3")
    val predictions_accumulator4 = new ListBufferAccumulator[(Double, Double, Double)]
    spark.sparkContext.register(predictions_accumulator4, "predictions4")
    val predictions_accumulator5 = new ListBufferAccumulator[(Double, Double, Double)]
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
          _._2.asInstanceOf[Double] >= 4.0
        ).map(_._1.asInstanceOf[Int]).toSet

        val user = recSys._matrix.rowIter.slice(userId - 1, userId).toList.head.toArray
        val selected = recSys.topKItemsForUser(user, userId)

        accumulator.add(
          new RankingMetrics(k = topK, selected, relevant).getRankingMetrics
        )
      }: Unit)

      val metricPerUser = accumulator.value
      val sumMetrics = metricPerUser.reduce((a, b) => {
        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
      })

      (
        sumMetrics._1 / metricPerUser.length,
        sumMetrics._2 / metricPerUser.length,
        sumMetrics._3 / metricPerUser.length
      )
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

  def hybridCrossValidation(spark: SparkSession, similarity: BaseSimilarity, topK: Int): Seq[Double] = {
    val recsys1 = new SequentialTopKRecommender().setGridSize(
      3, 3
    ).setNumberItems(1682).setMinParamsApriori(
      0.01, 0.9
    ).setMinParamsSequential(
      0.01, 0.9
    ).setPeriods(5).setK(topK)

    val recsys2 = new UserBasedTopKRecommender(25, topK)
    recsys2.setSimilarityMeasure(similarity)

    val hybrid = new HybridRecommenderTopK().setCF(
      recsys2
    ).setSequential(recsys1).setNumberOfItems(1682)

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

      hybrid.fit(train)

      val testData = test.groupBy("user_id").agg(
        collect_list(col("item_id")).as("items")
      ).collect()

      testData.foreach(row => {
        val userId = row.getInt(0)
        val items = row.getList(1).toArray()

        val relevant = items.map(_.asInstanceOf[Int]).toSet

        val selected = hybrid.transform(
          train.filter(col("user_id") === userId)
        )

        if (selected.isEmpty) {
          accumulator.add(0.0)
        } else {
          val evaluator = new TopKMetrics(k = topK, selected, relevant)
          accumulator.add(evaluator.getPrecision)
        }
      }: Unit)

      accumulator.value.sum / accumulator.value.length
    })
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master(
      "local[*]"
    ).config(
      "spark.sql.autoBroadcastJoinThreshold", "-1"
    ).appName(
      "TFM"
    ).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

/*
    val timePeriods = Seq(
      (0L, "1997-08-09 02:00:00.0", "1997-10-19 02:00:00.0"),
      (1L, "1997-10-19 02:00:00.0", "1997-12-29 01:00:00.0"),
      (2L, "1997-12-29 01:00:00.0", "1998-05-20 02:00:00.0")
    )
*/

    spark.stop()
  }
}
