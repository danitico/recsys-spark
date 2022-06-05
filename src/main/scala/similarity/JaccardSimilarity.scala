package similarity

class JaccardSimilarity extends BaseSimilarity {
  def getSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double = {
    val arrays = firstArray.zip(secondArray)

    val intersection = arrays.count(tuple => tuple._1 == tuple._2 && tuple._1 == 1 && tuple._2 == 1).toDouble
    val union = arrays.count(tuple => (tuple._1 + tuple._2) > 0).toDouble

    intersection / union
  }
}
