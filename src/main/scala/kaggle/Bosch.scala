package kaggle

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Dataset
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
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
    //val samplingFunc:Long=>Boolean = (Id => Id%1000<10) //only process one percent of the dataset
    val samplingFunc:Long=>Boolean = (Id => true) //all rows

    val (header, rdd) = getUnifiedDataset(spark, "file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/"    , samplingFunc )

    //print the schema of the rdd
    //println(header.mkString("|"))

    //print the 20 first entries
    //rdd.take(20).foreach(a => println("%d --> %s".format(a._1, a._2.mkString("|"))))

    //print number of entries that we consider
    //println("Number of entries considered is %d".format(rdd.count()))

//function bellow must be changed from dataset to rdd
    
    //compute per column statistics : Count, Mean, Variance, Min, Max
    //dsTrain.describe("Id", "Response").show()
    //dsTest.describe("Id", "Response").show()

    //check if we can build a single table  (no conflicting Id)
    //val mergedId =  ds.groupBy("Id").agg(countDistinct($"Response").name("distCnt"))
    //println("Number of entries considered after merged %d".format(mergedId.count()))

    //get all distinct signature by groupping all non null data together
    //ds.schema.filter(_.dataType != "Id" || _.dataType != "Response").map(_.name)
    //ds.show()

    val (headerLines, rddLines) = groupByStationBinary(header, rdd)
    println("Number of different patterns = %d".format(rddLines.count()))
    println("Lines = (%s)".format(headerLines.mkString(",")))
    rddLines.foreach(r => println("Pattern (%s) has %d entries".format(r._1.mkString(","), r._2)) )

    spark.stop()
  }

  def getUnifiedDataset(spark:SparkSession, inputDir:String, samplingFunc:Long => Boolean):Tuple2[Seq[String], RDD[Array[Any]] ] = {
    //load preprocess TRAIN dataset or create it
    val (dsTrainCatH, dsTrainCat) = getDataset(spark, inputDir+"train_categorical.csv", samplingFunc, ( _.toString) )
    val (dsTrainDatH, dsTrainDat) = getDataset(spark, inputDir+"train_date.csv"       , samplingFunc, ( _.toDouble) )
    val (dsTrainNumH, dsTrainNum) = getDataset(spark, inputDir+"train_numeric.csv"    , samplingFunc, ( _.toDouble) )

    val dsTrainH = dsTrainCatH ++ dsTrainDatH.slice(1, dsTrainDatH.length) ++ dsTrainNumH.slice(1, dsTrainNumH.length)
    val dsTrain = dsTrainCat.join(dsTrainDat).mapValues{case(left,right)=>left++right}.join(dsTrainNum).mapValues{case(left,right)=>left++right}

    //load preprocess TEST dataset or create it
    val (_, dsTestCat) = getDataset(spark, inputDir+"test_categorical.csv", samplingFunc, ( _.toString) )
    val (_, dsTestDat) = getDataset(spark, inputDir+"test_date.csv"       , samplingFunc, ( _.toDouble) )
    var (_, dsTestNum) = getDataset(spark, inputDir+"test_numeric.csv"    , samplingFunc, ( _.toDouble) )
    dsTestNum = dsTestNum.mapValues(r => r++Array(-1)) //Add negative response to the test sample

    val dsTest = dsTestCat.join(dsTestDat).mapValues{case(left,right)=>left++right}.join(dsTestNum).mapValues{case(left,right)=>left++right}

//    (dsTrainH, (dsTrain++dsTest).map(Array(_._1) ++ _._2))
    (dsTrainH, dsTrain.map(r => Array[Any](r._1) ++ r._2))
  }


  def getDataset(spark:SparkSession, file:String, samplingFunc:Long => Boolean, cast:String => Any):Tuple2[ Seq[String], RDD[Tuple2[Long,Array[Any]]] ] = {
      //read the header line and use it to infer the schema
      val header = spark.sparkContext.textFile(file).map(_.split(",", -1)).first()

      //return the dataset
      val rdd = spark.sparkContext.textFile(file).filter(!_.startsWith("Id")).map(_.split(",", -1) )
                     .filter(r => samplingFunc(r.head.toLong))
                     .map( r => (r.head.toLong, r.slice(1,r.length).map( s => if(s==""){null}else{cast(s)} ) ) )
      (header, rdd )
  }

  def groupByStationBinary(header:Seq[String], rdd:RDD[Array[Any]]):Tuple2[Seq[String], RDD[Tuple2[Seq[Int], Int]]] = {
    val format = Pattern.compile("L(\\d+)_S(\\d+)(_.*)");
    val groupColumns = header.zipWithIndex.map{ case (c, index) => 
       val m = format.matcher(c)
       if(m.find()){  ("L"+m.group(1)+"_S"+m.group(2) ,  Seq(index) ) }
       else        {  ("", Seq(index)) }
    }.groupBy(_._1).map{ case(key,values) => (key, values.map(a => a._2).reduce( (a,b) => a ++ b ) ) }.toSeq

    val binaryRdd = rdd.map{ row =>
       val key    = groupColumns.filter(_._1!="").map{ g => g._2.map(row(_)!=null).reduce( (a,b) => a||b) }.map(if(_){1}else{0})  //line_station info used as grouping key 
       val value = 1 //just count for now on
       (key, value)       
    }.reduceByKey{ (a,b) => a+b }
     .sortBy(_._2,false) //sort by decreasing count rate

    ( groupColumns.map(_._1) ,   binaryRdd)
  }

}


