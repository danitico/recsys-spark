package recommender.sequential

import accumulator.ListBufferAccumulator
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.ml.linalg.{SparseMatrix, Vectors}
import org.apache.spark.sql.functions.{col, collect_list, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable.ListBuffer

class TopKSequentialRecommender extends Serializable {
  private var _numberItems: Long = -1
  private var _kCustomers: Int = -1
  private var _kMeansDistance: String = "cosine"
  private var _assignedClusters: DataFrame = null
  private var _userItemDf: DataFrame = null
  private var _transactionDf: DataFrame = null
  private var _kMeansModel: KMeansModel = null
  private var _transactionGroups: Array[DataFrame] = null
  private var _transactionsModels: Array[KMeansModel] = null

  def setNumberItems(numberItems: Int): this.type = {
    this._numberItems = numberItems
    this
  }

  def setKCustomer(kCustomers: Int): this.type = {
    this._kCustomers = kCustomers
    this
  }

  def setKMeansDistance(distance: String): this.type = {
    this._kMeansDistance = distance
    this
  }

  def fit(train: DataFrame): Unit = {
    this._userItemDf = this.getUserItemDf(train)
    this.clusterCustomers()
    this._transactionDf = this.getTransactionDf(train)
    this.clusterTransactions()
    this.buildPeriods()
  }

  private def clusterCustomers(): Unit = {
    this._kMeansModel = new KMeans().setDistanceMeasure(
      this._kMeansDistance
    ).setK(
      this._kCustomers
    ).setFeaturesCol(
      "features"
    ).setPredictionCol(
      "customer_cluster"
    ).setMaxIter(10).fit(this._userItemDf)

    this._assignedClusters = this._kMeansModel.transform(
      this._userItemDf
    ).drop("features")
  }

  private def clusterTransactions(): Unit = {
    this._transactionDf = this._transactionDf.join(this._assignedClusters, "user_id")
    val clusters = this._kMeansModel.summary.cluster.select(
      "customer_cluster"
    ).distinct().orderBy(
      "customer_cluster"
    ).collect().map(_.getInt(0))

    // pruebo con Kmeans hasta resolver lo de SOM
    val transactionsGroupsAndModels = clusters.map(cluster_id => {
      val transactionGroup = this._transactionDf.where("customer_cluster == " + cluster_id)

      val model = new KMeans().setK(5).setPredictionCol("transaction_cluster").fit(transactionGroup)
      val transactionGroupWithPrediction = model.transform(transactionGroup)
      transactionGroupWithPrediction.show()

      (transactionGroupWithPrediction, model)
    })

    this._transactionGroups = transactionsGroupsAndModels.map(_._1)
    this._transactionsModels = transactionsGroupsAndModels.map(_._2)
  }

  private def buildPeriods(): Unit = {
    this._transactionGroups(0).show()
  }

  private def getNumberOfUsers(dataframe: DataFrame): Long = {
    dataframe.select("user_id").distinct().count()
  }

  private def getNotRepresentedItems(groupedDf: DataFrame): Seq[Int] = {
    val everyItem = Range.inclusive(1, this._numberItems.toInt).toSet
    val actualItems = groupedDf.select(
      "item_id"
    ).collect().map(_.getInt(0)).toSet

    everyItem.diff(actualItems).toSeq.sorted
  }

  private def createAndRegisterAccumulators: (ListBufferAccumulator[Long], ListBufferAccumulator[Long], ListBufferAccumulator[Double]) = {
    val session: SparkSession = SparkSession.getActiveSession.orNull

    val rowIndices = new ListBufferAccumulator[Long]
    val colSeparators = new ListBufferAccumulator[Long]
    val values = new ListBufferAccumulator[Double]

    session.sparkContext.register(rowIndices, "ratings")
    session.sparkContext.register(colSeparators, "col_separator")
    session.sparkContext.register(values, "row_indices")

    (rowIndices, colSeparators, values)
  }

  private def getUserItemDf(dataframe: DataFrame): DataFrame = {
    val session: SparkSession = SparkSession.getActiveSession.orNull
    import session.implicits._

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

    val numberOfUsers = this.getNumberOfUsers(dataframe)

    val sparse = new SparseMatrix(
      numRows = numberOfUsers.toInt,
      numCols = this._numberItems.toInt,
      colPtrs = separators.toArray.map(_.toInt),
      rowIndices = rowIndices.value.toArray.map(_.toInt),
      values = values.value.toArray
    )

    val matrixRows = sparse.toDense.rowIter.toSeq.map(_.toArray)

    val df = session.sparkContext.parallelize(
      matrixRows
    ).toDF(
      "features"
    ).withColumn("rowId1", monotonically_increasing_id())

    val ids = dataframe.select("user_id").distinct().orderBy("user_id").withColumn(
      "rowId2", monotonically_increasing_id()
    )

    df.join(ids, df("rowId1") === ids("rowId2"), "inner").drop("rowId1", "rowId2")
  }

  private def getTransactionDf(train: DataFrame): DataFrame = {
    val groupedByUserAndTimeStamp = train.groupBy("user_id", "timestamp").agg(
      collect_list(col("item_id")).as("items")
    )

    val arrayToBinary = udf((row: Array[Int]) => {
      Vectors.sparse(
        this._numberItems.toInt,
        row.map((element: Int) => {
          (element - 1, 1.0)
        })
      ).toDense
    })

    groupedByUserAndTimeStamp.withColumn(
      "features", arrayToBinary(groupedByUserAndTimeStamp("items"))
    ).drop("items")
  }
}
