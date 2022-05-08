package accumulators

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.ListBuffer

class DoubleListBufferAccumulator extends AccumulatorV2[Double, ListBuffer[Double]] {
  private val accumulator = ListBuffer[Double]()

  def add(element: Double): Unit = {
    this.accumulator += element
  }

  def copy(): DoubleListBufferAccumulator = {
    this
  }

  def isZero: Boolean = {
    this.accumulator.isEmpty
  }

  def merge(other: AccumulatorV2[Double, ListBuffer[Double]]): Unit = {
    this.accumulator.addAll(other.value)
  }

  def reset(): Unit = {
    this.accumulator.clear()
  }

  def value: ListBuffer[Double] = {
    this.accumulator
  }
}
