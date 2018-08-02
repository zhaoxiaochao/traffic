package com.meehoo.bigdata.crt.predict.supershortterm

import java.time.{LocalDateTime, LocalDate}
import java.time.format.DateTimeFormatter

import com.meehoo.bigdata.crt.base.indicator.station.PassengerIndex
import com.meehoo.bigdata.crt.common._
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FileSystem, Path}
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.mutable.ListBuffer

/**
  * lstm预测
  *
  */
class LSTMPredict(sparkSession: SparkSession, pGranule: String)
  extends java.io.Serializable{
  val spark = sparkSession
  val granule = pGranule //10分钟 15分钟 30分钟 60分钟 ten/fifteen/thirty/sixty

  val exampleLen = 30 //一个批次里面多少个样本


  def getTimeFlag(pSTime: String): String ={

    var timeFlag: String = ""
    granule match {
      case "ten" => {
        val minute = pSTime.substring(11, 13)
        val ten = minute.toInt / 10 * 10

        timeFlag = "%s%02d".format(pSTime.substring(0, 11), ten)
      }
      case "fifteen" => {
        val minute = pSTime.substring(11, 13)
        val ten = minute.toInt / 15 * 15

        timeFlag = "%s%02d".format( pSTime.substring(0, 11), ten)
      }
      case "thirty" => {
        val minute = pSTime.substring(11, 13)
        val ten = minute.toInt / 30 * 30

        timeFlag = "%s%02d".format(pSTime.substring(0, 11), ten)
      }
      case "sixty" => {
        val minute = pSTime.substring(11, 13)
        val ten = minute.toInt / 60 * 60

        timeFlag = "%s%02d".format(pSTime.substring(0, 11), ten)
      }
      case _ => {
        timeFlag = pSTime
      }
    }
    timeFlag
  }

  def getBeforeData(pDate: String): DataFrame = {
    val docName = G.base_indicator_station_passengerIndex
    val schema = PassengerIndex.getSchema()

    import spark.implicits._
    val trainData = MongoUtil.loadToDataFrame(spark, docName, schema).filter($"sDate" <= pDate)
      .withColumn("in", $"selfIn" + $"transferIn")
      .withColumn("out", $"selfOut" + $"transferOut")
      .select($"sTime", $"stationCode", $"in", $"out")
      .map(row => {
        (row.getAs[String]("stationCode"), getTimeFlag(row.getAs[String]("sTime")),
          row.getAs[Long]("in"), row.getAs[Long]("out"))
      }).groupBy($"_1", $"_2").agg("_3"->"sum", "_4"->"sum")
      .withColumnRenamed("_1", "station")
      .withColumnRenamed("_2", "sTime")
      .withColumnRenamed("sum(_3)", "inNum")
      .withColumnRenamed("sum(_4)", "outNum")

    trainData.toDF()
  }

  def predictOnSpark(modelType: String, valCol: String, futureNum: Int, endDate: String): DataFrame = {
    import spark.implicits._

    val inNumDF = getBeforeData(endDate)
      .map(r=>{
      val station = r.getAs[String]("station")
      val sTime = r.getAs[String]("sTime")
      val num = r.getAs[Long](valCol)
        (station, (sTime, num))
      }).groupByKey(_._1).flatMapGroups((key:String, rs: Iterator[(String, (String, Long))]) => {
      val result = new ListBuffer[Tuple3[String, ListBuffer[Long], String]]

      val array = rs.toArray.sortBy(_._2._1) //升序
      val inNumList = new ListBuffer[Long]
      var idx = array.length - 1 - exampleLen

      val endTime = array.apply(array.length - 1)._2._1
      while(idx < array.length && idx >= 0){
        val inNum = array.apply(idx)._2._2
        inNumList.append(inNum)
        idx = idx +1
      }

      //后30个数字直接进行预测,31值。
      result.append((key, inNumList, endTime))
      result
    })

    val result = inNumDF.rdd.map(r => {
      val pValueList = new ListBuffer[Double]

      val temp = Nd4j.create(r._2.length)
      var i = 0
      while(i< r._2.length){
        temp.putScalar(i, r._2.apply(i))
        i += 1
      }
      val mean : Double = Nd4j.mean(temp).getDouble(0)
      val std : Double = Nd4j.std(temp).getDouble(0)
      //归一化之后的值
      val initArray : INDArray = getInitArray(r._2, mean, std)

      val modelPath = getModelPath(modelType, r._1)
      val model = loadModel(modelPath)
      if(model != null){
        val output = model.rnnTimeStep(initArray)
        val predictValue = output.mul(std).add(mean)

        pValueList.append(predictValue.getDouble(predictValue.length()-1))
        var nextArray = output
        for(i <- 1 until futureNum){
          nextArray = model.rnnTimeStep(nextArray)
          pValueList.append(nextArray.mul(std).add(mean).getDouble(nextArray.length()-1))
        }
        model.rnnClearPreviousState()
      }
      (r._1, pValueList, r._3)
    }).toDF().withColumnRenamed("_1", "_id")
      .withColumnRenamed("_2", "values")
      .withColumnRenamed("_3", "before")

    result
  }

  def getModelPath(modelType: String, station: String): String ={
    val modelPath = String.format("%s/model/LSTM/%s/%s/%s.zip", Config.hadoop_master, modelType, granule, station)
    modelPath
  }

  def getInitArray(data : ListBuffer[Long], mean : Double, std : Double) : INDArray = {
    val grateBatchSize : Int = 1
    val getGrateTimeStep : Int = 30

    val initArray : INDArray = Nd4j.create(Array[Int](grateBatchSize,1,getGrateTimeStep), 'f')
    for (i<- 0 until  grateBatchSize){
      for(j <- 0 until  getGrateTimeStep){
        initArray.putScalar(Array[Int](0, 0, j), data.apply(j))
      }
    }
    initArray.sub(mean).div(std)
  }

  def loadModel(path : String): MultiLayerNetwork = {
    val conf : Configuration = new Configuration()
    val dstPath : Path = new Path(path)
    val fs : FileSystem= dstPath.getFileSystem(conf)

    var restoredNet : MultiLayerNetwork = null
    if(fs.exists(dstPath)){
      val inputStream : FSDataInputStream = fs.open(dstPath)
      restoredNet = ModelSerializer.restoreMultiLayerNetwork(inputStream)
      inputStream.close()
//      fs.close() //出现checkOpen异常先注释掉
    }

    restoredNet
  }
}

/**
  * 预测模式：model
  *   stationIn：站点进站量预测
  *   stationOut: 站点出站出站量
  * 预测值个数：num
  * 截止日期： 预测新数据 日期大于现有数据 20201231
  *          预测已有的数据 日期小现有数据 20161231
  * pGranule：时间粒度 //10分钟 15分钟 30分钟 60分钟 ten/fifteen/thirty/sixty
  */
object LSTMPredict {

  def main(args: Array[String]) {
    if (args.length < 4) {
      System.err.println("Usage: LSTMPredict <model> <num> <endDate> <granule>")
      System.exit(1)
    }

    var predictModel = "stationIn"
    var predictNum: Int = 1
    var endDate = "20501231"
    var pGranule = "ten"

    if (args.length == 4) {
      predictModel = args(0)
      predictNum = args(1).toInt
      endDate = args(2)
      pGranule = args(3)
    }

    val spark = SparkSession
      .builder
//      .config("spark.master", "local[*]")
      .getOrCreate()

    predictModel match {
      case "stationIn" => {
        val predictDf = stationInPredict(spark, predictNum, endDate, pGranule)
        MongoUtil.dataFrameSaveToMongo(predictDf, "predict_station_in_"+pGranule)
      }
      case "stationOut" => {
        val predictDf = stationOutPredict(spark, predictNum, endDate, pGranule)
        MongoUtil.dataFrameSaveToMongo(predictDf, "predict_station_out_"+pGranule)
      }
      case _ =>{
        println("trainModel参数错误！！！")
      }
    }

    //关闭sparksession
    spark.stop()
  }

  /**
    * 训练站点进站人数的lstm模型
    *
    * @param sparkSession
    * @param predictNum 未来多少个值
    * @param endDate 取样本数据的截止时间
    * @return
    */
  def stationInPredict(sparkSession: SparkSession, predictNum: Int, endDate: String, pGranule: String): DataFrame = {
    val lstmPredict = new LSTMPredict(sparkSession, pGranule)

    //值预测
    val in = lstmPredict.predictOnSpark("inNum", "inNum", predictNum, endDate)
    in.cache()

    import sparkSession.implicits._

    val df = in.flatMap(row =>{
      val result = new ListBuffer[Tuple4[String, String, String, Double]]
      val station = row.getAs[String]("_id")
      val valueList = row.getList(1)

      //解析日期
      val before = row.getAs[String]("before")
      var time = LocalDateTime.parse(before, DateTimeFormatter.ofPattern("yyyyMMdd HHmm"))

      if(!valueList.isEmpty){
        for(i <- 0 until valueList.size()){
          val minutes = granule2Int(pGranule)
          time = time.plusMinutes(minutes)
          val sTime = time.format(DateTimeFormatter.ofPattern("yyyyMMdd HHmm"))
          result.append((station+sTime, station, sTime, valueList.get(i)))
        }
      }

      result
    }).toDF().withColumnRenamed("_1", "_id")
      .withColumnRenamed("_2", "station")
      .withColumnRenamed("_3", "sTime")
      .withColumnRenamed("_4", "num")

    val station = MySqlUtil.getStation(sparkSession).select($"stationCode", $"stationName")
      .filter($"isOpen" === 1)

    df.join(station, $"station" === $"stationCode")
      .select($"_id", $"station", $"sTime", $"num", $"stationName")
  }

  /**
    * 训练站点出站人数的lstm模型
    *
    * @param sparkSession
    * @param predictNum 未来多少个值
    * @param endDate 取样本数据的截止时间
    * @return
    */
  def stationOutPredict(sparkSession: SparkSession, predictNum: Int, endDate: String, pGranule: String): DataFrame = {
    val lstmPredict = new LSTMPredict(sparkSession, pGranule)

    //值预测
    val out = lstmPredict.predictOnSpark("outNum", "outNum", predictNum, endDate)
    out.cache()

    import sparkSession.implicits._

    val df = out.flatMap(row =>{
      val result = new ListBuffer[Tuple4[String, String, String, Double]]
      val station = row.getAs[String]("_id")
      val valueList = row.getList(1)

      //解析日期
      val before = row.getAs[String]("before")
      var time = LocalDateTime.parse(before, DateTimeFormatter.ofPattern("yyyyMMdd HHmm"))

      if(!valueList.isEmpty){
        for(i <- 0 until valueList.size()){
          val minutes = granule2Int(pGranule)
          time = time.plusMinutes(minutes)
          val sTime = time.format(DateTimeFormatter.ofPattern("yyyyMMdd HHmm"))
          result.append((station+sTime, station, sTime, valueList.get(i)))
        }
      }

      result
    }).toDF().withColumnRenamed("_1", "_id")
      .withColumnRenamed("_2", "station")
      .withColumnRenamed("_3", "sTime")
      .withColumnRenamed("_4", "num")

    val station = MySqlUtil.getStation(sparkSession).select($"stationCode", $"stationName")
      .filter($"isOpen" === 1)

    df.join(station, $"station" === $"stationCode")
      .select($"_id", $"station", $"sTime", $"num", $"stationName")
  }

  def granule2Int(pGranule : String): Int = {
    var minutes = 0
    pGranule match {
      case "ten" => {
        minutes = 10
      }
      case "fifteen" => {
        minutes = 15
      }
      case "thirty" => {
        minutes = 30
      }
      case "sixty" => {
        minutes = 60
      }
      case _ => {
        minutes = 1
      }
    }
    minutes
  }


}