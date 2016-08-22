package kaggle


import java.lang._ 
import scala.reflect.runtime.universe.TypeTag
import scala.reflect.runtime.universe.TypeTag._
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration._
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import Numeric.Implicits._
import Ordering.Implicits._

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window
//import org.apache.spark.sql.hive.HiveContext

import org.apache.spark.ml.classification.{OneVsRest, LogisticRegression}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.random._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating


import org.apache.spark.ml._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation._

import java.lang.Math
import java.text.SimpleDateFormat
import java.util.concurrent.TimeUnit
import java.util.{Calendar, Date}
import java.util.HashMap

import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._


object Bosch {
//:Dataset<Row> = 
  def getDataset(sc:org.apache.spark.SparkContext, sqlContext:org.apache.spark.sql.SQLContext, file:String, dataType:DataType):DataFrame = {
      val header =sc.textFile(file).map(_.split(",")).first()
                    .map( name => { if(name=="Id"){StructField(name,IntegerType, false)}else{StructField(name, dataType, true)}  })

      val df = sqlContext.read
                         .schema(StructType(header))
//                       .option("header", "true") // Use first line of all files as header
                         .option("inferSchema", "false") // Automatically infer data types
                         .csv(file)

      df //return the dataframe
  }


  def main(args: Array[String]) {
        if (args.length > 1) {
          args.foreach{ println }
        }

        val conf = new SparkConf()
        conf.setAppName("KaggleBoschLoicQ")
        //conf.set("spark.driver.memory", "4g")
        //conf.set("spark.executor.memory", "2g")
        //conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") //to avoid serialization issues with smile
        //conf.set("spark.kryoserializer.buffer.max", "256mb");
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")

        val sqlContext = new org.apache.spark.sql.SQLContext(sc)
        import sqlContext.implicits._

        val df = getDataset(sc, sqlContext, "file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/train_numeric.csv", DoubleType).sample(true, 0.001)
        df.printSchema()
        
        val dfTrainNum = sqlContext.read
                           .option("header", "true") // Use first line of all files as header
                           .option("inferSchema", "true") // Automatically infer data types
                           .csv("file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/train_numeric.csv")
                           .sample(true, 0.001)
        dfTrainNum.printSchema()
        

        val dfTrainCat = sqlContext.read
                           .option("header", "true") // Use first line of all files as header
                           .option("inferSchema", "true") // Automatically infer data types
                           .csv("file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/train_categorical.csv")
                           .sample(true, 0.001)
        dfTrainCat.printSchema()

        val dfTrainDat = sqlContext.read
                           .option("header", "true") // Use first line of all files as header
                           .option("inferSchema", "true") // Automatically infer data types
                           .csv("file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/train_date.csv")
                           .sample(true, 0.001)
        dfTrainDat.printSchema()


//        val dfTrain = dfTrainNum.join(dfTrainCat, dfTrainNum("Id") === dfTrainCat("Id"))
        val dfTrainTmp = dfTrainNum.join(dfTrainCat, dfTrainNum("Id") === dfTrainCat("Id"))
        val dfTrain    = dfTrainTmp.join(dfTrainDat, dfTrainNum("Id") === dfTrainDat("Id"))
        dfTrain.printSchema()
 


  }
}

