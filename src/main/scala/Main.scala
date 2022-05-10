import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

import recommenders.item_based.ItemBased
import recommenders.user_based.UserBased
import similarity._

object Main {
  def dataset(session: SparkSession, filename: String): DataFrame = {
    session.read.options(
      Map("header" -> "true")
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

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("TFM").getOrCreate()

    val dataframe = dataset(spark, "train.csv")

    val recSysItemBased = new ItemBased(spark)
    recSysItemBased.readDataframe(dataframe, 1682)
    recSysItemBased.setSimilarityMeasure(new PearsonSimilarity)

    val targetItem = recSysItemBased.matrix.rowIter.slice(266, 267).toList.head.toArray

    println(recSysItemBased.predictionRatingItem(targetItem, 5))

    val recSysUserBased = new UserBased(spark)

    recSysUserBased.readDataframe(dataframe, 1682)
    recSysUserBased.setSimilarityMeasure(new PearsonSimilarity)

    val newUser = recSysUserBased.matrix.rowIter.slice(4, 5).toList.head.toArray

    println(recSysUserBased.predictionRatingItem(newUser, 267))

    spark.stop()
  }
}
