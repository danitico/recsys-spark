/*
  similarity/CosineSimilarity.scala
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
package similarity

import scala.math.{pow, sqrt}


class CosineSimilarity extends BaseSimilarity {
  def getSimilarity(firstArray: Array[Double], secondArray: Array[Double]): Double = {
    val numerator = firstArray.zip(secondArray).map {case (a, b) => a*b}.sum
    val denominator = sqrt(
      firstArray.map(pow(_, 2)).sum
    ) * sqrt(
      secondArray.map(pow(_, 2)).sum
    )

    numerator / denominator
  }
}
