/*
  metrics/PredictionMetrics.scala
  Copyright (C) 2022 Daniel Ranchal Parrado <danielranchal@correo.ugr.es>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>
*/
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
