package similarity

abstract class Similarity {
  def getSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double
}
