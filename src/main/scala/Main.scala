import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors

import similarity.PearsonSimilarity
import user_based.UserBased

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("TFM").getOrCreate()

    var recSys = new UserBased(spark)

    recSys.readDataset("train.csv")
    recSys.calculateDenseMatrix()
    recSys.setSimilarityMeasure(new PearsonSimilarity)

    val newUser = recSys.userItemMatrix.rowIter.slice(0, 1).toList.head.toArray

    println(recSys.predictionRatingItem(newUser, 38))
/*
    val indices = newUser.zipWithIndex.filter(_._1 > 0).map(_._2)

    val frequenciesItems = topKUsers.reduce((a, b) => {
      a.zip(b).map {case (c, d) => (if (c > 0) 1 else 0) + (if (d > 0) 1 else 0)}
    })

    indices.foreach(
      frequenciesItems(_) = 0.0
    )

    val topNItems = frequenciesItems.zipWithIndex.filter(_._1 > 0).sortWith(_._1 > _._1).take(5).map(_._2)

    topNItems.map(_ + 1).foreach(println(_))
*/
    spark.stop()
  }
}
