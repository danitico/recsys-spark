/*
  accumulator/ListBufferAccumulator.scala
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
package accumulator

import scala.collection.mutable.ListBuffer

import org.apache.spark.util.AccumulatorV2


class ListBufferAccumulator[A] extends AccumulatorV2[A, ListBuffer[A]] {
  private val accumulator = ListBuffer[A]()

  def add(element: A): Unit = {
    this.accumulator += element
  }

  def copy(): ListBufferAccumulator[A] = {
    this
  }

  def isZero: Boolean = {
    this.accumulator.isEmpty
  }

  def merge(other: AccumulatorV2[A, ListBuffer[A]]): Unit = {
    this.accumulator ++= other.value
  }

  def reset(): Unit = {
    this.accumulator.clear()
  }

  def value: ListBuffer[A] = {
    this.accumulator
  }
}
