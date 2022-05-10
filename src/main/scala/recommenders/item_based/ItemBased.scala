package recommenders.item_based

import scala.math.abs
import org.apache.spark.ml.linalg.{DenseMatrix, Vector}
import org.apache.spark.sql.{DataFrame, SparkSession}
import recommenders.BaseRecommender


class ItemBased(session: SparkSession) extends BaseRecommender(session){
  override protected def calculateDenseMatrix(dataframe: DataFrame, rows: Long, cols: Long): DenseMatrix = {
    super.calculateDenseMatrix(dataframe, rows, cols).transpose
  }

  def getKSimilarItems(targetItem: Array[Double], k: Int, user: Int): List[(Double, Vector)] = {
    val itemsWithRating = this.matrix.rowIter.filter(_(user) > 0).toList

    val correlations = itemsWithRating.map(
      f => this.similarity.getSimilarity(targetItem, f.toArray)
    )

    correlations.zip(itemsWithRating).sortWith(_._1 > _._1).take(k)
  }

  def ratingCalculation(topKItems: List[(Double, Vector)], ratingMean: Double, user: Int): Double = {
    val numerator = topKItems.map(a => {
      a._1 * a._2(user)
    }).sum

    val denominator = topKItems.map(_._1).reduce(abs(_) + abs(_))

    ratingMean + (numerator/denominator)
  }

  def predictionRatingItem(targetItem: Array[Double], user: Int): Double = {
    val topKItems = this.getKSimilarItems(targetItem, 25, user - 1)
    val ratingMean = targetItem.sum / targetItem.length

    this.ratingCalculation(topKItems, ratingMean, user - 1)
  }
}
