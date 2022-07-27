package metrics

import scala.math.{sqrt, pow, abs}

class PredictionMetrics extends Serializable {
  private var _errors: Array[Double] = Array()

  def this(errors: Array[Double]) = {
    this()
    this._errors = errors
  }

  def getPredictionMetrics: (Double, Double) = {
    (this.getRMSE, this.getMAE)
  }

  private def getRMSE: Double = {
    sqrt(
      this._errors.map(pow(_, 2)).sum / this._errors.length
    )
  }

  private def getMAE: Double = {
    this._errors.map(abs).sum / this._errors.length
  }
}
