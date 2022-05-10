package accumulators

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.ListBuffer

class LongListBufferAccumulator extends AccumulatorV2[Long, ListBuffer[Long]] {
  private val accumulator = ListBuffer[Long]()

  def add(element: Long): Unit = {
    this.accumulator += element
  }

  def copy(): LongListBufferAccumulator = {
    this
  }

  def isZero: Boolean = {
    this.accumulator.isEmpty
  }

  def merge(other: AccumulatorV2[Long, ListBuffer[Long]]): Unit = {
    this.accumulator.addAll(other.value)
  }

  def reset(): Unit = {
    this.accumulator.clear()
  }

  def value: ListBuffer[Long] = {
    this.accumulator
  }
}
