package recommender.sequential

import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.{col, collect_list, collect_set, datediff, max, min, monotonically_increasing_id, struct, udf, when, window}
import org.apache.spark.sql.DataFrame

import som.{SOM, SOMModel}
import com.github.nscala_time.time.Imports._

import recommender.BaseRecommender


class SequentialTopKRecommender(kRecommendedItems: Int, numberOfItems: Long) extends BaseRecommender(numberOfItems = numberOfItems) {
  private var _k: Int = kRecommendedItems
  private var _transactionDf: DataFrame = null
  private var _transactionArray: Array[(Int, Array[Double])] = null
  private var _transactionModel: SOMModel = null
  private var _periodRanges: Seq[(Long, String, String)] = null
  private var _durationPeriod: String = ""
  private var _periods: Seq[(Long, String, String)] = null
  private var _periodsIds: List[Long] = null
  private var _numberPeriods: Int = -1
  private var _rules: Array[(Array[String], Array[String], Double, Double)] = null
  private var _heightGridSom: Int = 5
  private var _widthGridSom: Int = 5
  private var _minSupportApriori: Double = 0.0
  private var _minConfidenceApriori: Double = 0.0
  private var _minSupportSequential: Double = 0.0
  private var _minConfidenceSequential: Double = 0.0

  def setKRecommendedItems(kRecommendedItems: Int): this.type = {
    this._k = kRecommendedItems
    this
  }

  def setGridSize(height: Int, width: Int): this.type = {
    this._heightGridSom = height
    this._widthGridSom = width
    this
  }

  def setMinParamsApriori(minSupport: Double, minConfidence: Double): this.type = {
    this._minSupportApriori = minSupport
    this._minConfidenceApriori = minConfidence
    this
  }

  def setMinParamsSequential(minSupport: Double, minConfidence: Double): this.type = {
    this._minSupportSequential = minSupport
    this._minConfidenceSequential = minConfidence
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

  override def fit(train: DataFrame): Unit = {
    // Get transactions per user and timestamp. Coding bought products as a binary vector
    this._transactionDf = this.getTransactionDf(train)

    // Assign each transaction a period id depending on the data time provided
    this.buildPeriods()

    // Cluster transactions of each customer segment using SOM
    this.clusterTransactions()

    // Generating sequential rules as in CMRULES
    this.obtainRules()

    // Convert transaction df to array to speedup prediction phase
    this._transactionArray = this._transactionDf.filter(
      col("period_id") === this._periodsIds.last
    ).select(
      "transaction_cluster", "features"
    ).collect().map(transaction => {
      (
        transaction.getInt(0),
        transaction.getList(1).toArray().map(_.asInstanceOf[Double])
      )
    })
  }

  override def transform(transactionsUser: DataFrame): Seq[(Int, Double)] = {
    val transactionDf = this.getTransactionDf(transactionsUser)
    val transactionDfWithPeriods = this.transformPeriods(transactionDf)
    val transactionsWithClusters = this._transactionModel.transform(
      transactionDfWithPeriods
    )

    val items = this.obtainItemsForTargetUser(transactionsWithClusters)
    val desiredConsequent = this.getMostAppropriateConsequent(items)

    if (desiredConsequent == -1) {
      Seq()
    } else {
      val transactionsAtT = this._transactionArray.filter(transaction => {
        transaction._1 == desiredConsequent
      })

      if (transactionsAtT.isEmpty) {
        Seq()
      } else {
        val previouslyItems = transactionsUser.select(
          "item_id"
        ).distinct().collect().map(_.getInt(0)).toSet

        val candidates = transactionsAtT.map(transaction => {
          transaction._2.zipWithIndex.filter(_._1 > 0.0).map(_._2 + 1)
        }).reduce((a: Array[Int], b: Array[Int]) => {
          a ++ b
        }).groupBy(identity).map(possibleCandidate => {
          (possibleCandidate._1, possibleCandidate._2.length.toDouble)
        }).filter(_._2 > 0).toSeq

        candidates.filter(candidate => {
          !previouslyItems.contains(candidate._1)
        }).sortWith(_._2 > _._2).take(
          this._k
        )
      }
    }
  }

  private def transformPeriods(transactions: DataFrame): DataFrame = {
    val timestampToPeriod = udf((timestamp: String) => {
      val formatInDataframe = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")
      val formatInPeriods = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss.S")
      val actualTimestamp = DateTime.parse(timestamp, formatInDataframe)

      val results = this._periods.map(range => {
        if (actualTimestamp >= DateTime.parse(range._2, formatInPeriods) && actualTimestamp < DateTime.parse(range._3, formatInPeriods)) {
          range._1
        } else {
          -1L
        }
      })

      if (results.forall(_ == -1L)) {
        // In case that a transaction is not assigned to a period, it is assigned to the last one by default
        this._periods.last._1
      } else {
        // assigning the id of the period
        results.filter(_ >= 0L).head
      }
    })

    transactions.withColumn(
      "period_id",
      timestampToPeriod(col("timestamp"))
    ).drop("timestamp")
  }

  private def obtainItemsForTargetUser(transactionsWithClusters: DataFrame): List[String] = {
    val flatList = udf((row: List[List[(Long, List[Int])]]) => {
      // getting list of sets being the first element the id of the period
      // and as second element the item with its period
      val generated = row.map(element => {
        (
          element.head._1,
          element.head._2.map(
            _.toString + "_" + (element.head._1 - this._periodsIds.length + 1).toString
          )
        )
      })

      // ordering list of sequence (returning [] if that period has not items)
      // and returning a flat map (representing a transaction for the rule extractor)
      this._periodsIds.map((id: Long) => {
        if(generated.exists(_._1 == id)) {
          generated.find(_._1 == id).orNull
        } else {
          (id, Seq())
        }
      }).sortWith(_._1 < _._1).flatMap(_._2)
    })

    transactionsWithClusters.groupBy("user_id", "period_id").agg(
      collect_set(col("transaction_cluster")).as("transaction_clusters")
    ).groupBy("user_id", "period_id").agg(
      collect_list(struct(col("period_id"), col("transaction_clusters"))).as("tuple_period_cluster")
    ).groupBy("user_id").agg(
      collect_list(col("tuple_period_cluster")).as("tuple_period_cluster_per_user")
    ).withColumn(
      "items",
      flatList(col("tuple_period_cluster_per_user"))
    ).select("items").first().getList(0).toArray.map(_.asInstanceOf[String]).toList
  }

  private def getMostAppropriateConsequent(items: List[String]): Int = {
    val score = this._rules.map(rule => {
      val similarity = rule._1.map(element => {
        if (items.contains(element)) 1 else 0
      }).sum.toDouble

      similarity * rule._3 * rule._4
    })

    val rulesWithScore = this._rules.zip(score).map {
      case ((a, b, c, d), e) => (a, b, c, d, e)
    }.filter(_._5 > 0)

    if (rulesWithScore.isEmpty) {
      -1
    } else {
      rulesWithScore.sortWith(
        _._5 > _._5
      ).head._2.head.split("_").head.toInt
    }
  }

  private def getTransactionDf(train: DataFrame): DataFrame = {
    // Group dataset by user and timestamp
    val groupedByUserAndTimeStamp = train.groupBy("user_id", "timestamp").agg(
      collect_list(col("item_id")).as("items")
    )

    // Creating udf to convert list of integers into a binary array
    val arrayToBinary = udf((row: Array[Int]) => {
      Vectors.sparse(
        this._numberOfItems.toInt,
        row.map((element: Int) => {
          (element - 1, 1.0)
        })
      ).toDense.toArray.toList
    })

    // Calling udf
    groupedByUserAndTimeStamp.withColumn(
      "features", arrayToBinary(groupedByUserAndTimeStamp("items"))
    ).drop("items")
  }

  private def buildPeriods(): Unit = {
    // Build periods depending on the input received
    // - ranges of datetimes provided
    // - duration of each period provided
    // - or desired number of periods

    if (this._periodRanges != null) {
      this.buildPeriodsFromProvidedRanges()
    } else if (this._durationPeriod.nonEmpty) {
      this.buildPeriodsFromDuration()
    } else if (this._numberPeriods > 0) {
      this.buildPeriodsFromNumberOfPartitions()
    }
  }

  private def buildPeriodsFromProvidedRanges(): Unit = {
    // udf to parse each timestamp to a given period depending on the datetime ranges
    val timestampToPeriod = udf((timestamp: String) => {
      val formatInDataframe = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss")
      val formatInPeriods = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss.S")
      val actualTimestamp = DateTime.parse(timestamp, formatInDataframe)

      val results = this._periodRanges.map(range => {
        if (actualTimestamp >= DateTime.parse(range._2, formatInPeriods) && actualTimestamp < DateTime.parse(range._3, formatInPeriods)) {
          range._1
        } else {
          -1L
        }
      })

      if (results.forall(_ == -1L)) {
        // In case that a transaction is not assigned to a period, it is assigned to the last one by default
        this._periodRanges.last._1
      } else {
        // assigning the id of the period
        results.filter(_ >= 0L).head
      }
    })

    // calling the udf
    this._transactionDf = this._transactionDf.withColumn(
      "period_id",
      timestampToPeriod(col("timestamp"))
    )

    this._periods = this._periodRanges

    // Getting the period ids
    this._periodsIds = this._periods.map(_._1).toList
  }

  private def buildPeriodsFromDuration(): Unit = {
    // Using window to divide by periods
    this._transactionDf = this._transactionDf.withColumn(
      "period",
      window(col("timestamp"), this._durationPeriod)
    )

    // getting the set of periods of generated and generating an id
    val periods = this._transactionDf.select(
      "period"
    ).distinct().orderBy("period").withColumn(
      "period_id",
      monotonically_increasing_id()
    )

    // changing the ranges generated by their id
    this._transactionDf = this._transactionDf.join(
      periods, this._transactionDf("period") === periods("period"), "inner"
    ).drop("period")

    // add start and end datetime
    this._periods = periods.withColumn(
      "start",
      col("period.start")
    ).withColumn(
      "end",
      col("period.end")
    ).drop("period").collect().toSeq.map(row => {
      (row.getLong(0), row.getTimestamp(1).toString, row.getTimestamp(2).toString)
    })

    // Getting period ids
    this._periodsIds = this._periods.map(_._1).toList
  }

  private def buildPeriodsFromNumberOfPartitions(): Unit = {
    // get the difference between the first and the last timestamp in days
    val diff = this._transactionDf.agg(
      min("timestamp").as("start"),
      max("timestamp").as("end")
    ).withColumn(
      "diff", datediff(col("end"), col("start"))
    ).head().getInt(2)

    // using window to establish the ranges using that difference and the number of periods
    this._transactionDf = this._transactionDf.withColumn(
      "period",
      window(col("timestamp"), (diff.toDouble / this._numberPeriods.toDouble).toInt.toString + " days")
    )

    // getting unique periods and assigning them an id
    var periods = this._transactionDf.select(
      "period"
    ).distinct().orderBy("period").withColumn(
      "period_id",
      monotonically_increasing_id()
    )

    // If the number of periods generated is greater than the ones desired
    // those extra periods are assigned to the last desired period
    if (periods.count() > this._numberPeriods) {
      periods = periods.withColumn(
        "period_id",
        when(
          col("period_id") > this._numberPeriods - 1, this._numberPeriods - 1
        ).otherwise(col("period_id"))
      )
    }

    // changing timestamps ranges by ids in the transaction dataframe
    this._transactionDf = this._transactionDf.join(
      periods, this._transactionDf("period") === periods("period"), "inner"
    ).drop("period")

    // getting start and end timestamp for each period
    this._periods = periods.withColumn(
      "start",
      col("period.start")
    ).withColumn(
      "end",
      col("period.end")
    ).drop("period").collect().toSeq.map(row => {
      (row.getLong(0), row.getTimestamp(1).toString, row.getTimestamp(2).toString)
    })

    // Getting period ids
    this._periodsIds = this._periods.map(_._1).distinct.toList
  }

  private def clusterTransactions(): Unit = {
    // a SOM is run to get the groups of transactions
    this._transactionModel = new SOM().setMaxIter(5).setHeight(
      this._heightGridSom
    ).setWidth(
      this._widthGridSom
    ).setFeaturesCol(
      "features"
    ).setPredictionCol(
      "transaction_cluster"
    ).setSeed(42L).fit(this._transactionDf)

    this._transactionDf = this._transactionModel.transform(this._transactionDf)
  }

  private def obtainRules(): Unit = {
    // udf to flatten list and add metadata to each item
    val flatList = udf((row: List[List[(Long, List[Int])]]) => {
      // getting list of sets being the first element the id of the period
      // and as second element the item with its period
      val generated = row.map(element => {
        (
          element.head._1,
          element.head._2.map(
            _.toString + "_" + (element.head._1 - this._periodsIds.length + 1).toString
          )
        )
      })

      // ordering list of sequence (returning [] if that period has not items)
      // and returning a flat map (representing a transaction for the rule extractor)
      this._periodsIds.map((id: Long) => {
        if(generated.exists(_._1 == id)) {
          generated.find(_._1 == id).orNull
        } else {
          (id, Seq())
        }
      }).sortWith(_._1 < _._1).flatMap(_._2)
    })

    // udf to filter rules which consequent does not have an item from the period 0 (actual period)
    val filterAntecedent = udf((row: Array[String]) => {
      row.filter(!_.endsWith("_0"))
    })

    // udf to get X union Y from X -> Y
    val getXY = udf((column1: List[String], column2: List[String]) => {
      column1 ++ column2
    })

    // get transactions per period for each user
    val transactions = this._transactionDf.groupBy("user_id", "period_id").agg(
      collect_set(col("transaction_cluster")).as("transaction_clusters")
    ).groupBy("user_id", "period_id").agg(
      collect_list(struct(col("period_id"), col("transaction_clusters"))).as("tuple_period_cluster")
    ).groupBy("user_id").agg(
      collect_list(col("tuple_period_cluster")).as("tuple_period_cluster_per_user")
    ).withColumn(
      "items",
      flatList(
        col("tuple_period_cluster_per_user")
      )
    ).drop("tuple_period_cluster_per_user").orderBy("user_id")

    // train fpgrowth model
    val fpgrowth = new FPGrowth().setItemsCol(
      "items"
    ).setMinSupport(
      this._minSupportApriori
    ).setMinConfidence(
      this._minConfidenceApriori
    )
    val model = fpgrowth.fit(transactions)

    // Remove from the antecedent those items belonging to period 0
    // and removing columns of metrics related to rules
    // because they were extracted from the original rules
    // After that, distinct rules are obtained
    val rules = model.associationRules.filter(row => {
      val consequent = row.getList(1).toArray()
      consequent.head.asInstanceOf[String].endsWith("_0")
    }).withColumn(
      "antecedent",
      filterAntecedent(col("antecedent"))
    ).filter(row => row.getList(0).toArray().nonEmpty).drop(
      "confidence", "lift", "support"
    ).distinct()

    // Solving support and confidence for the new set of rules
    val numberOfTransactions = transactions.count()
    val transactionsArray = transactions.select("items").collect().map(_.getList(0).toArray())

    // udf for support
    val getSupport = udf((row: List[String]) => {
      transactionsArray.map(transaction => {
        if (row.toSet.subsetOf(transaction.map(_.asInstanceOf[String]).toSet)) {
          1.0
        } else {
          0.0
        }
      }).sum / numberOfTransactions.toDouble
    })

    // rules with support and confidence
    val sequentialRules = rules.withColumn(
      "XY",
      getXY(col("antecedent"), col("consequent"))
    ).withColumn(
      "support",
      getSupport(col("XY"))
    ).withColumn(
      "support_antecedent",
      getSupport(col("antecedent"))
    ).withColumn(
      "confidence",
      col("support") / col("support_antecedent")
    ).drop("XY", "support_antecedent")

    // filter sequential rules with min support and confidence
    this._rules = sequentialRules.filter(
      col("support") > this._minSupportSequential
    ).filter(
      col("confidence") > this._minConfidenceSequential
    ).collect().map(rule => {
      (
        rule.getList(0).toArray().map(_.asInstanceOf[String]),
        rule.getList(1).toArray().map(_.asInstanceOf[String]),
        rule.getDouble(2),
        rule.getDouble(3)
      )
    })
  }
}
