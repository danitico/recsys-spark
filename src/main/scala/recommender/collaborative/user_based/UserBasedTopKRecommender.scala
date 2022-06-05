package recommender.collaborative.user_based

import org.apache.spark.ml.linalg.Vector
import recommender.collaborative.BaseRecommender

class UserBasedTopKRecommender(kUsers: Int, kItems: Int) extends BaseRecommender{
  protected var _kUsers: Int = kUsers
  protected var _kItems: Int = kItems

  def setKUsers(k: Int): Unit = {
    this._kUsers = k
  }

  def setKItems(k: Int): Unit = {
    this._kItems = k
  }

  protected def getKSimilarUsers(targetUser: Array[Double], item: Int): List[(Double, Vector)] = {
    val usersWithRating = this._matrix.rowIter.filter(_(item) > 0).toList

    if (usersWithRating.isEmpty) {
      return List()
    }

    val correlations = usersWithRating.map(
      f => this._similarity.getSimilarity(targetUser, f.toArray)
    )

    correlations.zip(usersWithRating).sortWith(_._1 > _._1).take(this._kUsers)
  }

  def topKItemsForUser(targetUser: Array[Double]): Set[Int] = {
    val unratedItems = targetUser.zipWithIndex.filter(_._1 == 0).map(_._2)

    unratedItems.map(item => {
      val similarUsers = this.getKSimilarUsers(targetUser, item)

      if (similarUsers.isEmpty) {
        (item + 1, 0.0)
      } else {
        val score = similarUsers.map(a => {
          a._1 * a._2(item)
        }).sum

        (item + 1, score)
      }
    }).sortWith(_._2 > _._2).take(this._kItems).map(_._1).toSet
  }
}
