/*
  recommender/BaseRecommender.scala
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
package recommender

import scala.collection.mutable.ListBuffer

import org.apache.spark.ml.linalg.{DenseMatrix, SparseMatrix}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, collect_list}

import accumulator.ListBufferAccumulator
import similarity.BaseSimilarity


class BaseRecommender(numberOfItems: Long, isUserBased: Boolean = true) extends RecommenderInterface(numberOfItems, isUserBased) {
  def getSimilarity: BaseSimilarity = this._similarity

  def setSimilarity(similarity: BaseSimilarity): Unit = {
    this._similarity = similarity
  }

  def getIsUserBased: Boolean = this._isUserBased

  def setIsUserBased(isUserBased: Boolean): Unit = {
    this._isUserBased = isUserBased
  }

  def getNumberOfItems: Long = this._numberOfItems

  def setNumberOfItems(numberOfItems: Long): Unit = {
    this._numberOfItems = numberOfItems
  }

  def fit(dataframe: DataFrame): Unit = {
    val numberOfUsers = this.getNumberOfUsers(dataframe)
    this._matrix = this.solveMatrix(dataframe, numberOfUsers)
  }

  def transform(target: Array[Double], index: Int): Double = -1.0

  def transform(target: Array[Double]): Seq[(Int, Double)] = Seq()

  def transform(target: DataFrame): Seq[(Int, Double)] = Seq()

  protected def getNumberOfUsers(dataframe: DataFrame): Long = {
    dataframe.select("user_id").distinct().count()
  }

  protected def getNotRepresentedItems(dataframeGroupByItem: DataFrame): Seq[Int] = {
    val everyItem = Range.inclusive(1, this._numberOfItems.toInt).toSet
    val actualItems = dataframeGroupByItem.select(
      "item_id"
    ).collect().map(_.getInt(0)).toSet

    everyItem.diff(actualItems).toSeq.sorted
  }

  protected def createAndRegisterAccumulators: (ListBufferAccumulator[Long], ListBufferAccumulator[Long], ListBufferAccumulator[Double]) = {
    val rowIndices = new ListBufferAccumulator[Long]
    val colSeparators = new ListBufferAccumulator[Long]
    val values = new ListBufferAccumulator[Double]

    val session: SparkSession = SparkSession.getActiveSession.orNull

    session.sparkContext.register(rowIndices, "ratings")
    session.sparkContext.register(colSeparators, "col_separator")
    session.sparkContext.register(values, "row_indices")

    (rowIndices, colSeparators, values)
  }

  protected def solveMatrix(dataframe: DataFrame, numberOfUsers: Long): DenseMatrix = {
    val groupedDf = dataframe.groupBy(
      "item_id"
    ).agg(
      collect_list(col("user_id")).as("users"),
      collect_list(col("rating")).as("ratings")
    ).drop("user_id", "rating")

    val notRepresentedItems = this.getNotRepresentedItems(groupedDf)
    val (rowIndices, colSeparators, values) = this.createAndRegisterAccumulators

    groupedDf.foreach((row: Row) => {
      val users = row.getList(1).toArray()
      val ratings = row.getList(2).toArray()

      users.zip(ratings).foreach(UserRatingTuple => {
        rowIndices.add(UserRatingTuple._1.asInstanceOf[Int] - 1)
        values.add(UserRatingTuple._2.asInstanceOf[Double])
      })

      colSeparators.add(values.value.length)
    })

    val separators: ListBuffer[Long] = 0.toLong +: colSeparators.value

    notRepresentedItems.foreach(index => {
      separators.insert(
        index - 1,
        separators(index - 1)
      )
    })

    val sparse = new SparseMatrix(
      numRows = numberOfUsers.toInt,
      numCols = this._numberOfItems.toInt,
      colPtrs = separators.toArray.map(_.toInt),
      rowIndices = rowIndices.value.toArray.map(_.toInt),
      values = values.value.toArray
    )

    if (this._isUserBased) {
      sparse.toDense
    } else {
      sparse.transpose.toDense
    }
  }
}
