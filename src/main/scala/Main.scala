import item_based.ItemBased
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import similarity.{CosineSimilarity, EuclideanSimilarity, PearsonSimilarity}
import user_based.UserBased

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("TFM").getOrCreate()

    val recSysItemBased = new ItemBased(spark)
    recSysItemBased.readDataset("train.csv")
    recSysItemBased.calculateDenseMatrix()
    recSysItemBased.setSimilarityMeasure(new PearsonSimilarity)

    val targetItem = recSysItemBased.itemUserMatrix.rowIter.slice(259, 260).toList.head.toArray

    println(recSysItemBased.predictionRatingItem(targetItem, 0))

    val recSysUserBased = new UserBased(spark)

    recSysUserBased.readDataset("train.csv")
    recSysUserBased.calculateDenseMatrix()
    recSysUserBased.setSimilarityMeasure(new EuclideanSimilarity)

    val newUser = recSysUserBased.userItemMatrix.rowIter.slice(0, 1).toList.head.toArray

    println(recSysUserBased.predictionRatingItem(newUser, 259))
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
