package accumulator

import org.apache.spark.util.AccumulatorV2

import scala.collection.mutable.ListBuffer

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
    this.accumulator.addAll(other.value)
  }

  def reset(): Unit = {
    this.accumulator.clear()
  }

  def value: ListBuffer[A] = {
    this.accumulator
  }
}
