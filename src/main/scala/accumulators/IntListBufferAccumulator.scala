package accumulators

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.ListBuffer

class IntListBufferAccumulator extends AccumulatorV2[Int, ListBuffer[Int]] {
  private val accumulator = ListBuffer[Int]()

  def add(element: Int): Unit = {
    this.accumulator += element
  }

  def copy(): IntListBufferAccumulator = {
    this
  }

  def isZero: Boolean = {
    this.accumulator.isEmpty
  }

  def merge(other: AccumulatorV2[Int, ListBuffer[Int]]): Unit = {
    this.accumulator.addAll(other.value)
  }

  def reset(): Unit = {
    this.accumulator.clear()
  }

  def value: ListBuffer[Int] = {
    this.accumulator
  }
}
