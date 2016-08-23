package kaggle

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.rdd._

import java.util.regex.Matcher
import java.util.regex.Pattern

/*
import java.lang._ 
import scala.reflect.runtime.universe.TypeTag
import scala.reflect.runtime.universe.TypeTag._
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.duration._
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import Numeric.Implicits._
import Ordering.Implicits._

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window


//import org.apache.spark.ml.classification.{OneVsRest, LogisticRegression}
//import org.apache.spark.ml.param.ParamMap
//import org.apache.spark.mllib.linalg._
//import org.apache.spark.mllib.clustering._
//import org.apache.spark.mllib.random._
//import org.apache.spark.mllib.recommendation.ALS
//import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
//import org.apache.spark.mllib.recommendation.Rating

//import org.apache.spark.ml._
//import org.apache.spark.ml.classification._
//import org.apache.spark.ml.feature._
//import org.apache.spark.ml.evaluation._

import java.lang.Math
import java.text.SimpleDateFormat
import java.util.concurrent.TimeUnit
import java.util.{Calendar, Date}
import java.util.HashMap
*/




object Bosch {

  // $example on:create_ds$
  // Note: Case classes in Scala 2.10 can support only up to 22 fields. To work around this limit,
  // you can use custom classes that implement the Product interface
  case class Person(name: String, age: Long)
  // $example off:create_ds$

  def main(args: Array[String]) {
    if (args.length > 1) {
      args.foreach{ println }
    }

    val spark = SparkSession
       .builder()
       .appName("KaggleBoschLoicQ appName")
       .config("spark.debug.maxToStringFields", "9999")
       .config("spark.driver.memory", "8g")
       .config("spark.executor.memory", "4g")
       .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") //to avoid serialization issues with smile
       .config("spark.kryoserializer.buffer.max", "256mb")
       .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
   
    // Import spark implicits for implicit conversions like converting RDDs to DataFrames
    // Must be call after the spark session is created
    import spark.implicits._



    //options
    val samplingRate = 1.0000 //only process a fraction of the dataset
 
    //load preprocess dataset or create it
    val dsTrain = try {  
       spark.read.parquet("temporaryData/joinedTrain")
    }catch{
       case _ : Throwable => {
          val dsTrainNum = getDataset(spark, "file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/train_numeric.csv"    , DoubleType)
          val dsTrainCat = getDataset(spark, "file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/train_categorical.csv", StringType)
          val dsTrainDat = getDataset(spark, "file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/train_date.csv"       , DoubleType)

          //join the three datasets in one
          val dsTrain = sortColumns(
                           dsTrainNum.join(dsTrainCat, dsTrainCat("Id") === dsTrainNum("Id")).drop(dsTrainCat("Id"))
                                     .join(dsTrainDat, dsTrainDat("Id") === dsTrainNum("Id")).drop(dsTrainDat("Id"))
                        )
         
          dsTrain.write.mode("overwrite").parquet("temporaryData/joinedTrain")
          dsTrain
       }
    }

    //print the schema of the joined dataset
    dsTrain.printSchema()

    //print the 20 first entries
    dsTrain.show()
 
    dsTrain.take(25).foreach(println)

    //print number of entries that we consider
    println("Number of entries considered is %d after %f sampling".format(dsTrain.count(), samplingRate))

    spark.stop()
  }

  def getDataset(spark:SparkSession, file:String, dataType:DataType):Dataset[Row] = {
      //read the header line and use it to infer the schema
      val schema = spark.sparkContext.textFile(file).map(_.split(",")).first()
                    .map( name => { if(name=="Id"){StructField(name,LongType, nullable = false)}else{StructField(name, dataType, nullable = true)}  })

      //return the dataset
      spark.sqlContext.read
                .schema(StructType(schema))
//                .option("nullValue", "-1")
                .option("inferSchema", "false") // do not automatically infer data types (very slow)
                .option("header", "true")
                .csv(file) 

  }

  def sortColumns(ds:Dataset[Row]):Dataset[Row] = {
    val format = Pattern.compile("L(\\d+)_S(\\d+)_(.)(\\d+)");

    val orderedColumns = ds.columns.sortWith( (a,b) => {
       val ma =   format.matcher(a)
       val mb =   format.matcher(b)
       if(ma.find()){
          if(mb.find()){
             if(ma.group(4)==mb.group(4)) ma.group(3)<mb.group(3)
             else ma.group(4).toInt<mb.group(4).toInt
          }else{
             false
          }
       }else{
          if(mb.find()){
             true
          }else{
             a<b
          }
       }
    })
    ds.select(orderedColumns.map(ds.col(_)):_*)
  }



/*
  //USED TO CREATE A DATASET DIRECTLY FROM A RDD[Row], but it seems to bug because the rows have too many column
  private def getRowRDD(spark:SparkSession, file:String, dataType:DataType):(RDD[Array[String]], Array[StructField]) = {
      //read the header line and use it to infer the schema
      val schemaRDD = spark.sparkContext.textFile(file).map(_.split(",",-1)).first()
                    .map( name => { if(name=="Id"){StructField(name,StringType, false)}else{StructField(name, dataType, true)}  })

      val rawRDD = spark.sparkContext.textFile(file).filter(!_.startsWith("Id")).map(_.split(",",-1))
      (rawRDD, schemaRDD)
  }

  private def getUnifiedDataset(spark:SparkSession, file:String, sampling:Double):Dataset[Row] = {
     val (inputRDD1, schema1) = getRowRDD(spark, file+"_numeric.csv"    , StringType)
     val (inputRDD2, schema2) = getRowRDD(spark, file+"_categorical.csv", StringType)
     val (inputRDD3, schema3) = getRowRDD(spark, file+"_date.csv"       , StringType)

     val schemaUni = schema1 ++ schema2.slice(1,schema2.length) ++ schema3.slice(1,schema3.length)

     val pairRDD1 = inputRDD1.map( a => (a.head, a                   ) )
     val pairRDD2 = inputRDD2.map( a => (a.head, a.slice(1,a.length) ) ) //drop the Id
     val pairRDD3 = inputRDD3.map( a => (a.head, a.slice(1,a.length) ) ) //drop the Id
     
     val inputRDDUni = pairRDD1.join(pairRDD2).join(pairRDD3).mapValues{ case(v1,v2) => ( v1._1++v1._2++v2) }.values //merged using key but return only values
     val rowRDDUni = inputRDDUni.sample(true, sampling).map(Row.fromSeq(_))//.cache()

     println(rowRDDUni.first.mkString(";"))

     spark.createDataFrame(rowRDDUni, StructType(schemaUni))      

  }
*/


}


