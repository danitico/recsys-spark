package metrics

class TopKMetrics extends Serializable {
  protected var _k: Int = 0
  protected var _relevantAndSelected: Int = 0
  protected var _Nis: Int = 0
  protected var _Nrn: Int = 0
  protected var _relevantSize: Int = 0

  def this(k: Int, selected: Set[Int], relevant: Set[Int]) = {
    this()
    this._k = k
    this._relevantSize = relevant.size
    this._relevantAndSelected = selected.intersect(relevant).size
  }

  def getPrecision: Double = {
    this._relevantAndSelected / this._k
  }

  def getRecall: Double = {
    this._relevantAndSelected / this._relevantSize
  }

  def getF1Score: Double = {
    val precision = this.getPrecision
    val recall = this.getRecall

    (2 * recall * precision) / (recall + precision)
  }
}
