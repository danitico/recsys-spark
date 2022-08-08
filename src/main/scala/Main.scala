import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, from_unixtime}
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

//  def userBasedCrossValidation(spark: SparkSession, similarity: BaseSimilarity, k: Int): Seq[(Double, Double)] = {
//    val recSysUserBased = new UserBasedRatingRecommender(k)
//    recSysUserBased.setSimilarityMeasure(similarity)
//
//    val predictions_accumulator1 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator1, "predictions1")
//    val predictions_accumulator2 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator2, "predictions2")
//    val predictions_accumulator3 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator3, "predictions3")
//    val predictions_accumulator4 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator4, "predictions4")
//    val predictions_accumulator5 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator5, "predictions5")
//
//    Seq(1, 2, 3, 4, 5).map(index => {
//      println("Fold " + index)
//      val train = dataset("data/train-fold" + index + ".csv")
//      val test = dataset("data/test-fold" + index + ".csv")
//
//      val accumulator = index match {
//        case 1 => predictions_accumulator1
//        case 2 => predictions_accumulator2
//        case 3 => predictions_accumulator3
//        case 4 => predictions_accumulator4
//        case 5 => predictions_accumulator5
//      }
//
//      recSysUserBased.fit(train, 1682)
//
//      test.groupBy("user_id").agg(
//        collect_list(col("item_id")).as("items"),
//        collect_list(col("rating")).as("ratings")
//      ).foreach(row => {
//        val userId = row.getInt(0)
//        val items = row.getList(1).toArray()
//        val ratings = row.getList(2).toArray()
//
//        val targetUser = recSysUserBased._matrix.rowIter.slice(userId - 1, userId).toList.head.toArray
//        items.zip(ratings).foreach(a => {
//          val prediction = recSysUserBased.transform(
//            targetUser, a._1.asInstanceOf[Int]
//          )
//
//          val difference = prediction - a._2.asInstanceOf[Double]
//          accumulator.add(difference)
//        }: Unit)
//      }: Unit)
//
//      val metrics = new PredictionMetrics(accumulator.value.toArray)
//
//      metrics.getPredictionMetrics
//    })
//  }
//
//  def itemBasedCrossValidation(spark: SparkSession, similarity: BaseSimilarity, k: Int): Seq[(Double, Double)] = {
//    val recSysItemBased = new ItemBasedRatingRecommender(k)
//    recSysItemBased.setSimilarityMeasure(similarity)
//
//    val predictions_accumulator1 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator1, "predictions1")
//    val predictions_accumulator2 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator2, "predictions2")
//    val predictions_accumulator3 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator3, "predictions3")
//    val predictions_accumulator4 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator4, "predictions4")
//    val predictions_accumulator5 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator5, "predictions5")
//
//    Seq(1, 2, 3, 4, 5).map(index => {
//      println("Fold " + index)
//      val train = dataset("data/train-fold" + index + ".csv")
//      val test = dataset("data/test-fold" + index + ".csv")
//
//      val accumulator = index match {
//        case 1 => predictions_accumulator1
//        case 2 => predictions_accumulator2
//        case 3 => predictions_accumulator3
//        case 4 => predictions_accumulator4
//        case 5 => predictions_accumulator5
//      }
//
//      recSysItemBased.fit(train, 1682)
//
//      test.groupBy("item_id").agg(
//        collect_list(col("user_id")).as("users"),
//        collect_list(col("rating")).as("ratings")
//      ).foreach(row => {
//        val itemId = row.getInt(0)
//        val users = row.getList(1).toArray()
//        val ratings = row.getList(2).toArray()
//
//        val targetItem = recSysItemBased._matrix.rowIter.slice(itemId - 1, itemId).toList.head.toArray
//        users.zip(ratings).foreach(a => {
//          val prediction = recSysItemBased.transform(
//            targetItem, a._1.asInstanceOf[Int]
//          )
//
//          val difference = prediction - a._2.asInstanceOf[Double]
//          accumulator.add(difference)
//        }: Unit)
//      }: Unit)
//
//      val metrics = new PredictionMetrics(accumulator.value.toArray)
//
//      metrics.getPredictionMetrics
//    })
//  }
//
//  def userBasedTopKCrossValidation(spark: SparkSession, similarity: BaseSimilarity, kUsers: Int, topK: Int): Seq[(Double, Double, Double)] = {
//    val recSys = new UserBasedTopKRecommender(kUsers, topK)
//    recSys.setSimilarityMeasure(similarity)
//
//    val predictions_accumulator1 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator1, "predictions1")
//    val predictions_accumulator2 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator2, "predictions2")
//    val predictions_accumulator3 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator3, "predictions3")
//    val predictions_accumulator4 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator4, "predictions4")
//    val predictions_accumulator5 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator5, "predictions5")
//
//    Seq(1, 2, 3, 4, 5).map(index => {
//      println("Fold " + index)
//      val train = dataset("data/train-fold" + index + ".csv")
//      val test = dataset("data/test-fold" + index + ".csv")
//
//      val accumulator = index match {
//        case 1 => predictions_accumulator1
//        case 2 => predictions_accumulator2
//        case 3 => predictions_accumulator3
//        case 4 => predictions_accumulator4
//        case 5 => predictions_accumulator5
//      }
//
//      recSys.fit(train, 1682)
//
//      test.groupBy("user_id").agg(
//        collect_list(col("item_id")).as("items"),
//        collect_list(col("rating")).as("ratings")
//      ).foreach(row => {
//        val userId = row.getInt(0)
//        val items = row.getList(1).toArray()
//        val ratings = row.getList(2).toArray()
//
//        val relevant = items.zip(ratings).filter(
//          _._2.asInstanceOf[Double] >= 4.0
//        ).map(_._1.asInstanceOf[Int]).toSet
//
//        val user = recSys._matrix.rowIter.slice(userId - 1, userId).toList.head.toArray
//        val selected = recSys.transform(user)
//
//        accumulator.add(
//          new RankingMetrics(k = topK, selected.map(_._1).toSet, relevant).getRankingMetrics
//        )
//      }: Unit)
//
//      val metricPerUser = accumulator.value
//      val sumMetrics = metricPerUser.reduce((a, b) => {
//        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
//      })
//
//      val finalMetrics = (
//        sumMetrics._1 / metricPerUser.length,
//        sumMetrics._2 / metricPerUser.length,
//        sumMetrics._3 / metricPerUser.length
//      )
//      println(finalMetrics)
//      finalMetrics
//    })
//  }
//
//  def itemBasedTopKCrossValidation(spark: SparkSession, similarity: BaseSimilarity, kItems: Int, topK: Int): Seq[(Double, Double, Double)] = {
//    val recSys = new ItemBasedTopKRecommender(kItems, topK)
//    recSys.setSimilarityMeasure(similarity)
//
//    val predictions_accumulator1 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator1, "predictions1")
//    val predictions_accumulator2 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator2, "predictions2")
//    val predictions_accumulator3 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator3, "predictions3")
//    val predictions_accumulator4 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator4, "predictions4")
//    val predictions_accumulator5 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator5, "predictions5")
//
//    Seq(1, 2, 3, 4, 5).map(index => {
//      println("Fold " + index)
//      val train = dataset("data/train-fold" + index + ".csv")
//      val test = dataset("data/test-fold" + index + ".csv")
//
//      val accumulator = index match {
//        case 1 => predictions_accumulator1
//        case 2 => predictions_accumulator2
//        case 3 => predictions_accumulator3
//        case 4 => predictions_accumulator4
//        case 5 => predictions_accumulator5
//      }
//
//      recSys.fit(train, 1682)
//
//      test.groupBy("user_id").agg(
//        collect_list(col("item_id")).as("items"),
//        collect_list(col("rating")).as("ratings")
//      ).foreach(row => {
//        val userId = row.getInt(0)
//        val items = row.getList(1).toArray()
//        val ratings = row.getList(2).toArray()
//
//        val relevant = items.zip(ratings).filter(
//          _._2.asInstanceOf[Double] >= 4.0
//        ).map(_._1.asInstanceOf[Int]).toSet
//
//        val user = recSys._matrix.colIter.slice(userId - 1, userId).toList.head.toArray
//        val selected = recSys.transform(user)
//
//        accumulator.add(
//          new RankingMetrics(k = topK, selected.map(_._1).toSet, relevant).getRankingMetrics
//        )
//      }: Unit)
//
//      val metricPerUser = accumulator.value
//      val sumMetrics = metricPerUser.reduce((a, b) => {
//        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
//      })
//
//      (
//        sumMetrics._1 / metricPerUser.length,
//        sumMetrics._2 / metricPerUser.length,
//        sumMetrics._3 / metricPerUser.length
//      )
//    })
//  }
//
//  def contentBasedTopKCrossValidation(spark: SparkSession, similarity: BaseSimilarity, kItems: Int, topK: Int): Seq[(Double, Double, Double)] = {
//    val recSys = new ContentBasedTopKRecommender(kItems, topK)
//    recSys.setSimilarityMeasure(similarity)
//
//    val features = spark.read.options(
//      Map("header" -> "true", "inferSchema" -> "true")
//    ).csv("data/features.csv").cache()
//
//    recSys.setFeatures(features)
//
//    val predictions_accumulator1 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator1, "predictions1")
//    val predictions_accumulator2 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator2, "predictions2")
//    val predictions_accumulator3 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator3, "predictions3")
//    val predictions_accumulator4 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator4, "predictions4")
//    val predictions_accumulator5 = new ListBufferAccumulator[(Double, Double, Double)]
//    spark.sparkContext.register(predictions_accumulator5, "predictions5")
//
//    Seq(1, 2, 3, 4, 5).map(index => {
//      println("Fold " + index)
//      val train = dataset("data/train-fold" + index + ".csv")
//      val test = dataset("data/test-fold" + index + ".csv")
//
//      val accumulator = index match {
//        case 1 => predictions_accumulator1
//        case 2 => predictions_accumulator2
//        case 3 => predictions_accumulator3
//        case 4 => predictions_accumulator4
//        case 5 => predictions_accumulator5
//      }
//
//      recSys.fit(train, 1682)
//
//      test.groupBy("user_id").agg(
//        collect_list(col("item_id")).as("items"),
//        collect_list(col("rating")).as("ratings")
//      ).foreach(row => {
//        val userId = row.getInt(0)
//        val items = row.getList(1).toArray()
//        val ratings = row.getList(2).toArray()
//
//        val relevant = items.zip(ratings).filter(
//          _._2.asInstanceOf[Double] >= 4.0
//        ).map(_._1.asInstanceOf[Int]).toSet
//
//        val user = recSys._matrix.colIter.slice(userId - 1, userId).toList.head.toArray
//        val selected = recSys.transform(user)
//
//        accumulator.add(
//          new RankingMetrics(k = topK, selected.map(_._1).toSet, relevant).getRankingMetrics
//        )
//      }: Unit)
//
//      val metricPerUser = accumulator.value
//      val sumMetrics = metricPerUser.reduce((a, b) => {
//        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
//      })
//
//      (
//        sumMetrics._1 / metricPerUser.length,
//        sumMetrics._2 / metricPerUser.length,
//        sumMetrics._3 / metricPerUser.length
//      )
//    })
//  }
//
//  def contentCrossValidation(spark: SparkSession, similarity: BaseSimilarity, k: Int): Seq[Double] = {
//    val recSys = new ContentBasedRatingRecommender(k)
//    recSys.setSimilarityMeasure(similarity)
//
//    val features = spark.read.options(
//      Map("header" -> "true", "inferSchema" -> "true")
//    ).csv("data/features.csv").cache()
//    recSys.setFeatures(features)
//
//    val predictions_accumulator1 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator1, "predictions1")
//    val predictions_accumulator2 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator2, "predictions2")
//    val predictions_accumulator3 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator3, "predictions3")
//    val predictions_accumulator4 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator4, "predictions4")
//    val predictions_accumulator5 = new ListBufferAccumulator[Double]
//    spark.sparkContext.register(predictions_accumulator5, "predictions5")
//
//    Seq(1, 2, 3, 4, 5).map(index => {
//      println("Fold " + index)
//      val train = dataset("data/train-fold" + index + ".csv")
//      val test = dataset("data/test-fold" + index + ".csv")
//
//      val accumulator = index match {
//        case 1 => predictions_accumulator1
//        case 2 => predictions_accumulator2
//        case 3 => predictions_accumulator3
//        case 4 => predictions_accumulator4
//        case 5 => predictions_accumulator5
//      }
//
//      recSys.fit(train, 1682)
//
//      test.groupBy("item_id").agg(
//        collect_list(col("user_id")).as("users"),
//        collect_list(col("rating")).as("ratings")
//      ).foreach(row => {
//        val itemId = row.getInt(0)
//        val users = row.getList(1).toArray()
//        val ratings = row.getList(2).toArray()
//
//        val itemFeature = recSys._features.filter(
//          _._1 == itemId
//        ).head._2
//        users.zip(ratings).foreach(a => {
//          val prediction = recSys.transform(
//            itemFeature, a._1.asInstanceOf[Int]
//          )
//
//          val difference = prediction - a._2.asInstanceOf[Double]
//          accumulator.add(difference)
//        }: Unit)
//      }: Unit)
//
//      sqrt(accumulator.value.map(pow(_, 2)).sum / accumulator.value.length)
//    })
//  }
//
//  def hybridCrossValidation(cf: ExplicitBaseRecommender, sequential: SequentialTopKRecommender, numberOfItems: Int, topK: Int): Seq[((Double, Double, Double), (Double, Double, Double))] = {
//    val spark = SparkSession.getActiveSession.orNull
//
//    val hybrid = new HybridRecommenderTopK().setCF(
//      cf
//    ).setSequential(
//      sequential
//    ).setNumberOfItems(
//      numberOfItems
//    )
//
//    val predictions_accumulator1 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator1, "predictions1")
//    val predictions_accumulator2 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator2, "predictions2")
//    val predictions_accumulator3 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator3, "predictions3")
//    val predictions_accumulator4 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator4, "predictions4")
//    val predictions_accumulator5 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator5, "predictions5")
//
//    Seq(1, 2, 3, 4, 5).map(index => {
//      println("Fold " + index)
//      val train = dataset("dbfs:/tfm/data/train-fold" + index + ".csv")
//      val test = dataset("dbfs:/tfm/data/test-fold" + index + ".csv")
//
//      val accumulator = index match {
//        case 1 => predictions_accumulator1
//        case 2 => predictions_accumulator2
//        case 3 => predictions_accumulator3
//        case 4 => predictions_accumulator4
//        case 5 => predictions_accumulator5
//      }
//
//      hybrid.fit(train)
//
//      val testData = test.groupBy("user_id").agg(
//        collect_list(col("item_id")).as("items"),
//        collect_list(col("rating")).as("ratings")
//      ).collect()
//
//      testData.foreach(row => {
//        val userId = row.getInt(0)
//        val items = row.getList(1).toArray()
//        val ratings = row.getList(2).toArray()
//
//        val relevant = items.zip(ratings).filter(
//          _._2.asInstanceOf[Double] >= 4.0
//        ).map(_._1.asInstanceOf[Int]).toSet
//
//        val selected = hybrid.transform(
//          train.filter(col("user_id") === userId)
//        )
//
//        accumulator.add(
//          (
//            new RankingMetrics(k = topK, selected._1.map(_._1).toSet, relevant).getRankingMetrics,
//            new RankingMetrics(k = topK, selected._2.map(_._1).toSet, relevant).getRankingMetrics
//          )
//        )
//      }: Unit)
//
//      val metricPerUserCF = accumulator.value.map(_._1)
//      val metricPerUserHybrid = accumulator.value.map(_._2)
//
//      val sumMetricsCF = metricPerUserCF.reduce((a, b) => {
//        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
//      })
//
//      val sumMetricsHybrid = metricPerUserHybrid.reduce((a, b) => {
//        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
//      })
//
//      val finalMetricsCF = (
//        sumMetricsCF._1 / metricPerUserCF.length,
//        sumMetricsCF._2 / metricPerUserCF.length,
//        sumMetricsCF._3 / metricPerUserCF.length
//      )
//
//      val finalMetricsHybrid = (
//        sumMetricsHybrid._1 / metricPerUserHybrid.length,
//        sumMetricsHybrid._2 / metricPerUserHybrid.length,
//        sumMetricsHybrid._3 / metricPerUserHybrid.length
//      )
//
//      println(finalMetricsCF)
//      println(finalMetricsHybrid)
//      (finalMetricsCF, finalMetricsHybrid)
//    })
//  }
//
//  def hybridContentCrossValidation(contentRecsys: ContentBasedTopKRecommender, sequential: SequentialTopKRecommender, numberOfItems: Int, topK: Int): Seq[((Double, Double, Double), (Double, Double, Double))] = {
//    val spark = SparkSession.getActiveSession.orNull
//
//    val hybrid = new HybridContentRecommenderTopK().setCF(
//      contentRecsys
//    ).setSequential(
//      sequential
//    ).setNumberOfItems(
//      numberOfItems
//    )
//
//    val predictions_accumulator1 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator1, "predictions1")
//    val predictions_accumulator2 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator2, "predictions2")
//    val predictions_accumulator3 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator3, "predictions3")
//    val predictions_accumulator4 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator4, "predictions4")
//    val predictions_accumulator5 = new ListBufferAccumulator[((Double, Double, Double), (Double, Double, Double))]
//    spark.sparkContext.register(predictions_accumulator5, "predictions5")
//
//    Seq(1, 2, 3, 4, 5).map(index => {
//      println("Fold " + index)
//      val train = dataset("dbfs:/FileStore/shared_uploads/tfm/data/train_fold" + index + ".csv")
//      val test = dataset("dbfs:/FileStore/shared_uploads/tfm/data/test_fold" + index + ".csv")
//
//      val accumulator = index match {
//        case 1 => predictions_accumulator1
//        case 2 => predictions_accumulator2
//        case 3 => predictions_accumulator3
//        case 4 => predictions_accumulator4
//        case 5 => predictions_accumulator5
//      }
//
//      hybrid.fit(train)
//
//      val testData = test.groupBy("user_id").agg(
//        collect_list(col("item_id")).as("items"),
//        collect_list(col("rating")).as("ratings")
//      ).collect()
//
//      testData.foreach(row => {
//        val userId = row.getInt(0)
//        val items = row.getList(1).toArray()
//        val ratings = row.getList(2).toArray()
//
//        val relevant = items.zip(ratings).filter(
//          _._2.asInstanceOf[Double] >= 4.0
//        ).map(_._1.asInstanceOf[Int]).toSet
//
//        val selected = hybrid.transform(
//          train.filter(col("user_id") === userId)
//        )
//
//        accumulator.add(
//          (
//            new RankingMetrics(k = topK, selected._1.map(_._1).toSet, relevant).getRankingMetrics,
//            new RankingMetrics(k = topK, selected._2.map(_._1).toSet, relevant).getRankingMetrics
//          )
//        )
//      }: Unit)
//
//      val metricPerUserContent = accumulator.value.map(_._1)
//      val metricPerUserHybrid = accumulator.value.map(_._2)
//
//      val sumMetricsContent = metricPerUserContent.reduce((a, b) => {
//        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
//      })
//
//      val sumMetricsHybrid = metricPerUserHybrid.reduce((a, b) => {
//        (a._1 + b._1, a._2 + b._2, a._3 + b._3)
//      })
//
//      val finalMetricsContent = (
//        sumMetricsContent._1 / metricPerUserContent.length,
//        sumMetricsContent._2 / metricPerUserContent.length,
//        sumMetricsContent._3 / metricPerUserContent.length
//      )
//
//      val finalMetricsHybrid = (
//        sumMetricsHybrid._1 / metricPerUserHybrid.length,
//        sumMetricsHybrid._2 / metricPerUserHybrid.length,
//        sumMetricsHybrid._3 / metricPerUserHybrid.length
//      )
//
//      println(finalMetricsContent)
//      println(finalMetricsHybrid)
//      (finalMetricsContent, finalMetricsHybrid)
//    })
//  }

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

    println("hola")

    spark.stop()
  }
}
