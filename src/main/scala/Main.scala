import item_based.ItemBased
import org.apache.spark.sql.SparkSession
import similarity.{CosineSimilarity, EuclideanSimilarity, PearsonSimilarity}
import user_based.UserBased

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[*]").appName("TFM").getOrCreate()

    val recSysItemBased = new ItemBased(spark)
    recSysItemBased.readDataset("train.csv", 1682)
    recSysItemBased.calculateDenseMatrix()
    recSysItemBased.setSimilarityMeasure(new PearsonSimilarity)

    val targetItem = recSysItemBased.itemUserMatrix.rowIter.slice(367, 368).toList.head.toArray

    println(recSysItemBased.predictionRatingItem(targetItem, 4))

    val recSysUserBased = new UserBased(spark)

    recSysUserBased.readDataset("train.csv", 1682)
    recSysUserBased.calculateDenseMatrix()
    recSysUserBased.setSimilarityMeasure(new PearsonSimilarity)

    val newUser = recSysUserBased.userItemMatrix.rowIter.slice(4, 5).toList.head.toArray

    println(recSysUserBased.predictionRatingItem(newUser, 368))

    spark.stop()
  }
}
