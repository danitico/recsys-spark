/*
  Main.scala
  Copyright (C) 2022 Daniel Ranchal Parrado <danielranchal@correo.ugr.es>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>
*/
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list, from_unixtime}
import accumulator.ListBufferAccumulator
import metrics.RankingMetrics
import recommender.BaseRecommender
import recommender.collaborative.item_based.ItemBasedTopKRecommender
import recommender.collaborative.user_based.UserBasedTopKRecommender
import recommender.content.ContentBasedTopKRecommender
import recommender.hybrid.HybridRecommenderTopK
import recommender.sequential.SequentialTopKRecommender
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

  def datasetFeatures(filename: String): DataFrame = {
    val session = SparkSession.getActiveSession.orNull

    session.read.options(
      Map("header" -> "true", "inferSchema" -> "true")
    ).csv(filename)
  }

  def userBasedTopKCrossValidation(similarity: BaseSimilarity, kUsers: Int, topK: Int, numberOfItems: Long): Seq[(Double, Double, Double)] = {
    val spark = SparkSession.getActiveSession.orNull

    val recSys = new UserBasedTopKRecommender(kUsers, topK, numberOfItems)
    recSys.setSimilarity(similarity)

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

      recSys.fit(train)

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
        val selected = recSys.transform(user)

        accumulator.add(
          new RankingMetrics(k = topK, selected.map(_._1).toSet, relevant).getRankingMetrics
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

      finalMetrics
    })
  }
  def itemBasedTopKCrossValidation(similarity: BaseSimilarity, kItems: Int, topK: Int, numberOfItems: Long): Seq[(Double, Double, Double)] = {
    val spark = SparkSession.getActiveSession.orNull

    val recSys = new ItemBasedTopKRecommender(kItems, topK, numberOfItems)
    recSys.setSimilarity(similarity)

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

      recSys.fit(train)

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

        val user = recSys._matrix.colIter.slice(userId - 1, userId).toList.head.toArray
        val selected = recSys.transform(user)

        accumulator.add(
          new RankingMetrics(k = topK, selected.map(_._1).toSet, relevant).getRankingMetrics
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

      finalMetrics
    })
  }
  def contentBasedTopKCrossValidation(similarity: BaseSimilarity, kItems: Int, topK: Int, numberOfItems: Long): Seq[(Double, Double, Double)] = {
    val spark = SparkSession.getActiveSession.orNull

    val recSys = new ContentBasedTopKRecommender(kItems, topK, numberOfItems)
    recSys.setSimilarity(similarity)

    val features = datasetFeatures("data/features.csv")

    recSys.setFeatures(features)

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

      recSys.fit(train)

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

        val user = recSys._matrix.colIter.slice(userId - 1, userId).toList.head.toArray
        val selected = recSys.transform(user)

        accumulator.add(
          new RankingMetrics(k = topK, selected.map(_._1).toSet, relevant).getRankingMetrics
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

      finalMetrics
    })
  }

  def hybridCrossValidation(firstRecSys: BaseRecommender, secondRecSys: BaseRecommender, weightFirstRecSys: Double, weightSecondRecSys: Double, topK: Int, numberOfItems: Int): Seq[(Double, Double, Double)] = {
    val spark = SparkSession.getActiveSession.orNull

    val hybrid = new HybridRecommenderTopK(topK, numberOfItems).setFirstRecommender(
      firstRecSys
    ).setSecondRecommender(
      secondRecSys
    ).setWeightFirstRecommender(weightFirstRecSys).setWeightSecondRecommender(weightSecondRecSys)

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

      hybrid.fit(train)

      val testData = test.groupBy("user_id").agg(
        collect_list(col("item_id")).as("items"),
        collect_list(col("rating")).as("ratings")
      ).collect()

      testData.foreach(row => {
        val userId = row.getInt(0)
        val items = row.getList(1).toArray()
        val ratings = row.getList(2).toArray()

        val relevant = items.zip(ratings).filter(
          _._2.asInstanceOf[Double] >= 4.0
        ).map(_._1.asInstanceOf[Int]).toSet

        val selected = hybrid.transform(
          train.filter(col("user_id") === userId)
        )

        accumulator.add(
          new RankingMetrics(k = topK, selected.map(_._1).toSet, relevant).getRankingMetrics,
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

      finalMetrics
    })
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master(
      "local[*]"
    ).config(
      "spark.sql.autoBroadcastJoinThreshold", "-1"
    ).config(
      "spark.jars", "lib/sparkml-som_2.12-0.2.1.jar"
    ).appName(
      "TFM"
    ).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    println("Try here your recommender")
//    User based approach
//    val results = userBasedTopKCrossValidation(new PearsonSimilarity, 25, 5, 1682)
//    println(results)

//    Item based approach
//    val results = itemBasedTopKCrossValidation(new CosineSimilarity, 25, 5, 1682)
//    println(results)

//    Content based approach
//    val results = contentBasedTopKCrossValidation(new EuclideanSimilarity, 25, 5, 1682)
//    println(results)

//    Hybrid approaches
//    val recSys1 = new UserBasedTopKRecommender(25, 5, 1682)
//    recSys1.setSimilarity(new PearsonSimilarity)

//    val recSys1 = new ItemBasedTopKRecommender(25, 5, 1682)
//    recSys1.setSimilarity(new PearsonSimilarity)

//    val recSys1 = new ContentBasedTopKRecommender(25, 5, 1682)
//    recSys1.setSimilarity(new PearsonSimilarity)

//    val recSys2 = new SequentialTopKRecommender(5, 1682).setGridSize(
//      3, 3
//    ).setPeriods(5).setMinParamsApriori(
//      0.01, 0.95
//    ).setMinParamsSequential(
//      0.01, 0.95
//    )
//    val results = hybridCrossValidation(recSys1, recSys2, 0.6, 0.4, 5, 1682)
//    println(result)

    spark.stop()
  }
}
