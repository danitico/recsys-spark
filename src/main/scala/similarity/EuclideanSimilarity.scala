package similarity

import scala.math.{pow, sqrt}

class EuclideanSimilarity extends Similarity {
  def getSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double = {
    // sum one to the denominator in order to avoid division by zero

    1 / (sqrt(firstArray.zip(secondArray).map {case (a, b) => pow(a - b, 2)}.sum) + 1)
  }
}
