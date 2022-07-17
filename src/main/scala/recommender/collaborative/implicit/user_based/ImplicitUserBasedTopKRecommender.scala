package recommender.collaborative.`implicit`.user_based

import org.apache.spark.ml.linalg.Vector
import recommender.collaborative.`implicit`.ImplicitBaseRecommender

class ImplicitUserBasedTopKRecommender(kUsers: Int, kItems: Int) extends ImplicitBaseRecommender{
  protected var _kUsers: Int = kUsers
  protected var _kItems: Int = kItems

  def setKUsers(k: Int): Unit = {
    this._kUsers = k
  }

  def setKItems(k: Int): Unit = {
    this._kItems = k
  }

  protected def getKSimilarUsers(targetUser: Array[Double]): List[Vector] = {
    val users = this._matrix.rowIter.toList

    val correlations = users.map(
      f => this._similarity.getSimilarity(targetUser, f.toArray)
    )

    correlations.zip(users).sortWith(_._1 > _._1).map(_._2).take(this._kUsers)
  }

  def transform(targetUser: Array[Double]): Array[(Int, Double)] = {
    val similarUsers = this.getKSimilarUsers(targetUser)

    val sumVectors = similarUsers.map(_.toArray).reduce(
      (x, y) => x.zip(y).map{case (a, b) => a + b}
    )

    val itemsFrequency = sumVectors.zipWithIndex.filter(
      _._1 > 0
    )

    val selectedItems = itemsFrequency.sortWith(
      _._1 > _._1
    )

    val maxFrequency = selectedItems.head._1

    selectedItems.map(item => {
      (item._2 + 1, item._1 / maxFrequency)
    })
  }
}
