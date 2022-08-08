package recommender.collaborative.user_based

import recommender.BaseRecommender


class UserBasedTopKRecommender(kUsers: Int, kItems: Int, numberOfItems: Long) extends BaseRecommender(numberOfItems = numberOfItems, isUserBased = true) {
  protected var _kUsers: Int = kUsers
  protected var _kItems: Int = kItems
  protected var _candidates: List[Array[Double]] = null

  def setKUsers(k: Int): Unit = {
    this._kUsers = k
  }

  def setKItems(k: Int): Unit = {
    this._kItems = k
  }

  protected def getKSimilarUsers(targetUser: Array[Double], item: Int): List[(Double, Array[Double])] = {
    val usersWithRating = this._candidates.filter(_(item) > 0)

    if (usersWithRating.isEmpty) {
      return List()
    }

    val correlations = usersWithRating.map(
      f => this._similarity.getSimilarity(targetUser, f)
    )

    correlations.zip(usersWithRating).sortWith(_._1 > _._1).take(this._kUsers)
  }

  override def transform(targetUser: Array[Double]): Seq[(Int, Double)] = {
    this._candidates = this._matrix.rowIter.map(_.toArray).toList

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
    }).sortWith(_._2 > _._2).take(this._kItems).toSeq
  }
}
