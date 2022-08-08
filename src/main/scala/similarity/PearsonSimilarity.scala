package similarity

import scala.math.{pow, sqrt}


class PearsonSimilarity extends BaseSimilarity {
  def getSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double = {
    val mean1 = firstArray.sum / firstArray.length
    val mean2 = secondArray.sum / secondArray.length

    val differencesFirstArray = firstArray.map(_ - mean1)
    val differencesSecondArray = secondArray.map(_ - mean2)
    val differencesFirstArraySquared = differencesFirstArray.map(pow(_, 2))
    val differencesSecondArraySquared = differencesSecondArray.map(pow(_, 2))

    val numerator = differencesFirstArray.zip(differencesSecondArray).map {case (a, b) => a * b}.sum
    val denominator = sqrt(differencesFirstArraySquared.sum) * sqrt(differencesSecondArraySquared.sum)

    numerator/denominator
  }
}
