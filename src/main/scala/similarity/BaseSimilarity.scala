package similarity

abstract class BaseSimilarity {
  def getSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double
}
