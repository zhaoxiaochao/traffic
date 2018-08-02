package com.meehoo.bigdata.crt.model.supershortterm

import com.meehoo.bigdata.crt.base.indicator.station.PassengerIndex
import com.meehoo.bigdata.crt.common.{Config, G, MongoUtil}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataOutputStream, FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import scala.collection.mutable.ListBuffer

/**
  *
  * @param sparkSession
  * @param pStartDate
  * @param pGranule //10分钟 15分钟 30分钟 60分钟 ten/fifteen/thirty/sixty
  */
class LSTMModel(sparkSession: SparkSession, pStartDate: String, pGranule: String)
  extends java.io.Serializable{
  val spark = sparkSession
  val startDate = pStartDate
  val granule = pGranule //10分钟 15分钟 30分钟 60分钟 ten/fifteen/thirty/sixty

  val epochs = 10 //模型重复多少次
  val batchSize = 3 //三个批次为一组
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

  def getTrainData(pDate: String): DataFrame = {
    val docName = G.base_indicator_station_passengerIndex
    val schema = PassengerIndex.getSchema()

    import spark.implicits._
    val trainData = MongoUtil.loadToDataFrame(spark, docName, schema).filter($"sDate" >= pDate)
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

  /**
    *
    * @param valCol inNum/outNum
    * @return
    */
  def trainOnSpark(valCol: String):RDD[(String, MultiLayerNetwork)] = {
    import spark.implicits._

    val inNumDF = getTrainData(startDate)
//        .filter($"station" === "0319")
      .map(r=>{
        val station = r.getAs[String]("station")
        val sTime = r.getAs[String]("sTime")
        val num = r.getAs[Long](valCol)
        (station, (sTime, num))
      }).groupByKey(_._1).flatMapGroups((key:String, rs: Iterator[(String, (String, Long))]) => {
      val result = new ListBuffer[Tuple2[String, ListBuffer[Long]]]

      val array = rs.toArray.sortBy(_._2._1) //升序
      val inNumList = new ListBuffer[Long]
      var idx = 0

      while(idx < array.length){
        val inNum = array.apply(idx)._2._2
        inNumList.append(inNum)
        idx = idx +1
      }
      result.append((key, inNumList))

      result
    })

    val modelList = inNumDF.rdd.map(r => {
      val station = r._1
      val meanAndStd = getMeanAndStd(r._2)
      val inNumLen = r._2.size

      //定义模型
      val model = createModel()
      //重复训练，每次重新设置所有值
      for(i <- 1 to epochs){//10
        //1到70，最后一个batch 70-99，共30个数字
        for(j <- 1 to inNumLen-exampleLen-batchSize){//100-30
        var input = Nd4j.create(Array[Int](batchSize, 1, exampleLen), 'f')
          var label = Nd4j.create(Array[Int](batchSize, 1, exampleLen), 'f')

          //3个样本一起训练，反向传播效果好一些，可以1个
          for(k <- 1 to batchSize){//3
            for(l <- 1 to exampleLen){//30
            var inputValue : Double = 0
              var labelValue : Double= 0

              inputValue = r._2.apply(j-1 + l-1 + k-1)
              labelValue = r._2.apply(j-1+1 + l-1 + k-1)

              input.putScalar(Array[Int](k-1, 0, l-1), inputValue)//curData
              label.putScalar(Array[Int](k-1, 0, l-1), labelValue)//nextData

            }
          }

          input = input.sub(meanAndStd._1).div(meanAndStd._2)
          label = label.sub(meanAndStd._1).div(meanAndStd._2)

          val dataSet = new DataSet(input, label)
          model.fit(dataSet)

          input.cleanup()
          label.cleanup()
        }
      }
      (station, model)
    })

    modelList
  }

  def saveOnHadoop(modelList: RDD[(String, MultiLayerNetwork)], modelType: String){
    //因为站点数据不多,所以可以不用Memory_and_disk方式
    modelList.cache()
    modelList.foreach(r=>saveModel(r._2, r._1, modelType))
  }

  /**
    * 创建lstm模型
    *
    * @return
    */
  def createModel(): MultiLayerNetwork = {
    val inputSize : Int = 1
    val outputSize : Int = 1
    val layer1Size : Int = 50
    val layer2Size : Int = 100
    val learningRate : Double = 0.006

    val conf : MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .learningRate(learningRate)
      .seed(123)
      .regularization(true)
      .l2(0.001)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.RMSPROP)   //To configure: .updater(new RmsProp(0.95))
      .list()
      .layer(0, new GravesLSTM.Builder().nIn(inputSize).nOut(layer1Size).activation(Activation.TANH).build())
      .layer(1, new GravesLSTM.Builder().nIn(layer1Size).nOut(layer2Size).activation(Activation.TANH).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
        .nIn(layer2Size).nOut(outputSize).build())//输出是三维的时间序列{numExamples, outputSize, sequenceLength}输入什么样的输出也是什么样的。
      .pretrain(false).backprop(true)
      .build()
    val net = new MultiLayerNetwork(conf)
    net.init()

    net
  }


  /**
    * 保存模型到hdfs对应的路径
    *
    * @param net
    * @param station
    */
  def saveModel(net : MultiLayerNetwork, station: String, modelType: String): Unit ={
    val conf : Configuration = new Configuration()
    val path = String.format("%s/model/LSTM/%s/%s/%s.zip", Config.hadoop_master, modelType, granule, station)
    val dstPath : Path = new Path(path)
    val fs : FileSystem = dstPath.getFileSystem(conf)
    val outputStream : FSDataOutputStream= fs.create(dstPath)
    val saveUpdater = true
    ModelSerializer.writeModel(net, outputStream, saveUpdater)
    outputStream.close()
    fs.close()
  }

  /**
    * 获取序列的均值和标准差
    *
    * @param input
    * @return
    */
  def getMeanAndStd(input: ListBuffer[Long]): Tuple2[Double, Double] = {
    val data = Nd4j.create(input.size)
    var i = 0
    while (i < input.size) {
      data.putScalar(i, input.apply(i))
      i += 1
    }
    return Tuple2(Nd4j.mean(data).getDouble(0), Nd4j.std(data).getDouble(0))
  }
}

/**
  * 输入参数
  * trainModel：训练模型
  *   stationIn：进站模型
  *   stationOut:出站模型
  * pStartDate: 日期开始时间
  * pGranule：时间粒度 //10分钟 15分钟 30分钟 60分钟 ten/fifteen/thirty/sixty
  */
object LSTMModel{

  def main(args : Array[String]): Unit ={
    if (args.length < 3) {
      System.err.println("Usage: LSTMModel <model> <startDate> <granule>")
      System.exit(1)
    }

    var predictModel = "stationIn"
    var pStartDate = "20180215"
    var pGranule = "ten"

    if (args.length == 3) {
      predictModel = args(0)
      pStartDate = args(1)
      pGranule = args(2)
    }

    val spark = SparkSession
      .builder
//      .config("spark.master", "local[*]")
      .getOrCreate()

    predictModel match {
      case "stationIn" => {
        stationInModel(spark, pStartDate, pGranule)
      }
      case "stationOut" => {
        stationOutModel(spark, pStartDate, pGranule)
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
    * @param pStartDate
    * @param pGranule
    */
  def stationInModel(sparkSession: SparkSession, pStartDate: String, pGranule: String): Unit = {

    val lstmModelTrain = new LSTMModel(sparkSession, pStartDate, pGranule)

    //训练模型
    val modelList = lstmModelTrain.trainOnSpark("inNum")
    lstmModelTrain.saveOnHadoop(modelList, "inNum")
  }

  /**
    * 训练站点进站人数的lstm模型
    *
    * @param sparkSession
    * @param pStartDate
    * @param pGranule
    */
  def stationOutModel(sparkSession: SparkSession, pStartDate: String, pGranule: String): Unit = {
    val lstmModelTrain = new LSTMModel(sparkSession, pStartDate, pGranule)
    lstmModelTrain.getTrainData(pStartDate).show()

    //训练模型
    val modelList = lstmModelTrain.trainOnSpark("outNum")
    lstmModelTrain.saveOnHadoop(modelList, "outNum")
  }
}