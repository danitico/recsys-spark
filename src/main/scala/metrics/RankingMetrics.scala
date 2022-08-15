/*
  metrics/RankingMetrics.scala
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


class RankingMetrics extends Serializable {
  private var _k: Int = 5
  private var _selected: Set[Int] = Set()
  private var _relevant: Set[Int] = Set()

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
    if (this._relevant.isEmpty) {
      0.0
    } else {
      this._selected.intersect(this._relevant).size / this._relevant.size.toDouble
    }
  }

  private def getAveragePrecision: Double = {
    Range.inclusive(1, this._k).map(i => {
      this._selected.take(i).intersect(this._relevant).size.toDouble / i.toDouble
    }).sum / this._k.toDouble
  }
}
