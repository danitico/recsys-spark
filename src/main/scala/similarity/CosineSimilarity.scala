package similarity

import scala.math.{pow, sqrt}

class CosineSimilarity extends BaseSimilarity {
  def getSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double = {
    val numerator = firstArray.zip(secondArray).map {case (a, b) => a*b}.sum
    val denominator = sqrt(
      firstArray.map(pow(_, 2)).sum
    ) * sqrt(
      secondArray.map(pow(_, 2)).sum
    )

    numerator / denominator
  }
}
