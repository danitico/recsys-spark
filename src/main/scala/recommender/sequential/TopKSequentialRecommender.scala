package recommender.sequential

import accumulator.ListBufferAccumulator
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.functions.{col, collect_list, collect_set, datediff, max, min, monotonically_increasing_id, struct, udf, when, window}
import org.apache.spark.ml.linalg.{SparseMatrix, Vectors}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import som.{SOM, SOMModel}
import com.github.nscala_time.time.Imports._

import scala.collection.mutable.ListBuffer

class TopKSequentialRecommender extends Serializable {
  private var _numberItems: Long = -1
  private var _kCustomers: Int = -1
  private var _kMeansDistance: String = "cosine"
  private var _assignedClusters: DataFrame = null
  private var _userItemDf: DataFrame = null
  private var _transactionDf: DataFrame = null
  private var _customerKmeansModel: KMeansModel = null
  private var _transactionGroups: Array[DataFrame] = null
  private var _transactionsModels: Array[SOMModel] = null
  private var _periodRanges: Seq[(Long, String, String)] = null
  private var _durationPeriod: String = ""
  private var _periods: DataFrame = null
  private var _periodsIds: List[Long] = null
  private var _numberPeriods: Int = -1

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

  def setPeriods(periods: Seq[(Long, String, String)]): this.type = {
    this._periodRanges = periods
    this
  }

  def setPeriods(duration: String): this.type = {
    this._durationPeriod = duration
    this
  }

  def setPeriods(numberPeriods: Int): this.type = {
    this._numberPeriods = numberPeriods
    this
  }

  def fit(train: DataFrame): Unit = {
    this._userItemDf = this.getUserItemDf(train)
    this.clusterCustomers()
    this._transactionDf = this.getTransactionDf(train)
    this.buildPeriods()
    this.clusterTransactions()
    this.obtainItemsets()
  }

  private def clusterCustomers(): Unit = {
    this._customerKmeansModel = new KMeans().setDistanceMeasure(
      this._kMeansDistance
    ).setK(
      this._kCustomers
    ).setFeaturesCol(
      "features"
    ).setPredictionCol(
      "customer_cluster"
    ).setMaxIter(10).setSeed(42L).fit(this._userItemDf)

    this._assignedClusters = this._customerKmeansModel.transform(
      this._userItemDf
    ).drop("features")
  }

  private def clusterTransactions(): Unit = {
    this._transactionDf = this._transactionDf.join(this._assignedClusters, "user_id")
    val clusters = this._assignedClusters.select(
      "customer_cluster"
    ).distinct().orderBy(
      "customer_cluster"
    ).collect().map(_.getInt(0))

    val transactionsGroupsAndModels = clusters.map(cluster_id => {
      val transactionGroup = this._transactionDf.where("customer_cluster == " + cluster_id)

      val model = new SOM().setMaxIter(10).setFeaturesCol(
        "features"
      ).setPredictionCol(
        "transaction_cluster"
      ).setSeed(42L).fit(transactionGroup)
      val transactionGroupWithPrediction = model.transform(transactionGroup)

      (transactionGroupWithPrediction, model)
    })

    this._transactionGroups = transactionsGroupsAndModels.map(_._1)
    this._transactionsModels = transactionsGroupsAndModels.map(_._2)
  }

  private def buildPeriods(): Unit = {
    if (this._periodRanges != null) {
      val timestampToPeriod = udf((timestamp: String) => {
        val format = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")
        val actualTimestamp = DateTime.parse(timestamp, format)
        val results = this._periodRanges.map(range => {
          if (actualTimestamp >= DateTime.parse(range._2, format) && actualTimestamp < DateTime.parse(range._3, format)) {
            range._1
          } else {
            -1L
          }
        })

        if (results.forall(_ == -1L)) {
          this._periodRanges.last._1
        } else {
          results.filter(_ >= 0L).head
        }
      })

      this._transactionDf = this._transactionDf.withColumn(
        "period_id",
        timestampToPeriod(col("timestamp"))
      )

      val session: SparkSession = SparkSession.getActiveSession.orNull
      import session.implicits._

      this._periods = this._periodRanges.toDF().withColumnRenamed(
        "_1",
        "period_id"
      ).withColumnRenamed(
        "_2",
        "start"
      ).withColumnRenamed(
        "_3",
        "end"
      )

      this._periodsIds = this._periodRanges.map(_._1).toList

    } else if (this._durationPeriod.nonEmpty) {
      this._transactionDf = this._transactionDf.withColumn(
        "period",
        window(col("timestamp"), this._durationPeriod)
      )

      this._periods = this._transactionDf.select(
        "period"
      ).distinct().orderBy("period")

      this._periods = this._periods.withColumn(
        "period_id",
        monotonically_increasing_id()
      )

      this._periodsIds = this._periods.select("period_id").orderBy("period_id").collect().map(
        _.getLong(0)
      ).toList

      this._transactionDf = this._transactionDf.join(
        this._periods, this._transactionDf("period") === this._periods("period"), "inner"
      ).drop("period")

      this._periods = this._periods.withColumn(
        "start",
        col("period.start")
      ).withColumn(
        "end",
        col("period.end")
      ).drop("period")
    } else if (this._numberPeriods > 0) {
      val diff = this._transactionDf.agg(
        min("timestamp").as("start"),
        max("timestamp").as("end")
      ).withColumn(
        "diff", datediff(col("end"), col("start"))
      ).head().getInt(2)

      this._transactionDf = this._transactionDf.withColumn(
        "period",
        window(col("timestamp"), (diff.toDouble / this._numberPeriods.toDouble).toInt.toString + " days")
      )

      this._periods = this._transactionDf.select(
        "period"
      ).distinct().orderBy("period")

      val periodsGenerated = _periods.count()

      this._periods = this._periods.withColumn(
        "period_id",
        monotonically_increasing_id()
      )

      this._periods = this._periods.withColumn(
        "period_id",
        when(
          col("period_id") === periodsGenerated - 1, periodsGenerated - 2
        ).otherwise(col("period_id"))
      )

      this._periodsIds = this._periods.select("period_id").distinct().orderBy("period_id").collect().map(
        _.getLong(0)
      ).toList

      this._transactionDf = this._transactionDf.join(
        this._periods, this._transactionDf("period") === this._periods("period"), "inner"
      ).drop("period")

      this._periods = this._periods.withColumn(
        "start",
        col("period.start")
      ).withColumn(
        "end",
        col("period.end")
      ).drop("period")
    }
  }

  private def obtainItemsets(): Unit = {
    val flatList = udf((row: List[List[(Long, List[Int])]]) => {
      val generated = row.map(element => {
        (element.head._1, element.head._2)
      })

      this._periodsIds.map((id: Long) => {
        if(generated.exists(_._1 == id)) {
          generated.find(_._1 == id).orNull
        } else {
          (id, Seq())
        }
      }).sortWith(_._1 < _._1).map(_._2)
    })

    val meow = this._transactionGroups(1).groupBy("user_id", "period_id").agg(
      collect_set(col("transaction_cluster")).as("transaction_clusters")
    ).groupBy("user_id", "period_id").agg(
      collect_list(struct(col("period_id"), col("transaction_clusters"))).as("tuple_period_cluster")
    ).groupBy("user_id").agg(
      collect_list(col("tuple_period_cluster")).as("tuple_period_cluster_per_user")
    ).withColumn(
      "sequence",
      flatList(
        col("tuple_period_cluster_per_user")
      )
    ).drop("tuple_period_cluster_per_user").orderBy("user_id")

    meow.show(false)

/*
    new PrefixSpan().setMinSupport(
      0.2
    ).setMaxPatternLength(
      5
    ).findFrequentSequentialPatterns(meow).show(false)
*/
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
      ).toDense.toArray.toList
    })

    groupedByUserAndTimeStamp.withColumn(
      "features", arrayToBinary(groupedByUserAndTimeStamp("items"))
    ).drop("items")
  }
}
