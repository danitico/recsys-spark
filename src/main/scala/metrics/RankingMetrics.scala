package metrics

class RankingMetrics extends Serializable {
  protected var _k: Int = 5
  protected var _selected: Set[Int] = Set()
  protected var _relevant: Set[Int] = Set()

  def this(k: Int, selected: Set[Int], relevant: Set[Int]) = {
    this()
    this._k = k
    this._selected = selected
    this._relevant = relevant
  }

  def getRankingMetrics: (Double, Double, Double) = {
    (this.getPrecision, this.getRecall, this.getAveragePrecision)
  }

  private def getPrecision: Double = {
    this._selected.intersect(this._relevant).size.toDouble / this._k.toDouble
  }

  private def getRecall: Double = {
    this._selected.intersect(this._relevant).size / this._relevant.size.toDouble
  }

  private def getAveragePrecision: Double = {
    Range.inclusive(1, this._k).map(i => {
      this._selected.take(i).intersect(this._relevant).size.toDouble / i.toDouble
    }).sum / this._k.toDouble
  }
}
