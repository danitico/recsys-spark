package similarity

abstract class BaseSimilarity extends Serializable {
  def getSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double
}
