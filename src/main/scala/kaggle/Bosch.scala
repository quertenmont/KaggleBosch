package kaggle

import scala.reflect.ClassTag

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.Dataset
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.rdd._
import org.apache.spark.util._

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
    //val samplingFunc:Long=>Boolean = (Id => Id%1000<1) //only process one percent of the dataset
    val samplingFunc:Long=>Boolean = (Id => true) //all rows

    val (header, rdd) = getUnifiedDataset(spark, "file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/"    , samplingFunc )


    //Caching will keep the dataset in memory (faster), but uses a lot of RAM
    //So that work only if you have really a lot of RAM on your PC or if you use a cluster
    //rdd.cache()

    //print the schema of the rdd
    //println(header.mkString("|"))

    //print the 20 first entries
    //rdd.take(20).foreach(a => println("%d --> %s".format(a._1, a._2.mkString("|"))))

    //print number of entries that we consider
    //println("Number of entries considered is %d".format(rdd.count()))

    //check if we can build a single table  (no conflicting Id)
    //println("Number of overlapping Id = %d".format(rdd.map(row => (row(0),1)).reduceByKey( _+_ ).filter(_._2 > 1).count))

    //compute per column statistics : Count, Mean, Variance, Min, Max
    printStat(spark, header, rdd)

    //get all distinct signature by groupping all non null data together
    val (headerLines, rddLines) = groupByStationBinary(header, rdd)
    println("Number of different patterns = %d".format(rddLines.count()))
    println("Lines = (%s)".format(headerLines.mkString(",")))
    rddLines.foreach(r => println("Pattern (%s) has %s entries".format(r._1, r._2.mkString(";"))) )

    //check all different types of category that we can get for each categorical feature
    //WARNING this is quite time consumming
    var categories = listPossibleCategories(header, rdd)
    println("category size = %d".format(categories.length))
    categories.foreach(r => println("%s --> %s".format(r._1.name, r._2.mkString(";") )) )


    spark.stop()
  }

  def getUnifiedDataset(spark:SparkSession, inputDir:String, samplingFunc:Long => Boolean):Tuple2[Seq[StructField], RDD[Array[Any]] ] = {
    //load preprocess TRAIN dataset or create it
    val (dsTrainCatH, dsTrainCat) = getDataset(spark, inputDir+"train_categorical.csv", samplingFunc, StringType, new MetadataBuilder().putString("type", "C").build )
    val (dsTrainDatH, dsTrainDat) = getDataset(spark, inputDir+"train_date.csv"       , samplingFunc, DoubleType, new MetadataBuilder().putString("type", "D").build )
    val (dsTrainNumH, dsTrainNum) = getDataset(spark, inputDir+"train_numeric.csv"    , samplingFunc, DoubleType, new MetadataBuilder().putString("type", "N").build )

    val dsTrainH = dsTrainCatH ++ dsTrainDatH.slice(1, dsTrainDatH.length) ++ dsTrainNumH.slice(1, dsTrainNumH.length)
    val dsTrain  = dsTrainCat.join(dsTrainDat).mapValues{case(left,right)=>left++right}.join(dsTrainNum).mapValues{case(left,right)=>left++right}

    //load preprocess TEST dataset or create it
    val (_, dsTestCat) = getDataset(spark, inputDir+"test_categorical.csv", samplingFunc, StringType, new MetadataBuilder().putString("type", "C").build  )
    val (_, dsTestDat) = getDataset(spark, inputDir+"test_date.csv"       , samplingFunc, DoubleType, new MetadataBuilder().putString("type", "D").build )
    var (_, dsTestNum) = getDataset(spark, inputDir+"test_numeric.csv"    , samplingFunc, DoubleType, new MetadataBuilder().putString("type", "N").build )
    dsTestNum = dsTestNum.mapValues(r => r++Array(-1)) //Add negative response to the test sample

    val dsTest = dsTestCat.join(dsTestDat).mapValues{case(left,right)=>left++right}.join(dsTestNum).mapValues{case(left,right)=>left++right}

//    (fillMetadataAndSort(dsTrainH), (dsTrain++dsTest).map(Array(_._1) ++ _._2))
    (fillMetadataAndSort(dsTrainH), dsTrain.map(r => Array[Any](r._1) ++ r._2))
  }

  def getDataset(spark:SparkSession, file:String, samplingFunc:Long => Boolean, dataType:DataType, metadata:Metadata):Tuple2[ Seq[StructField], RDD[Tuple2[Long,Array[Any]]] ] = {
      val cast:String=>Any = { dataType match {
         case _:DoubleType => (_.toDouble)
         case _            => (_.toString)
      }}
      
      //read the header line and use it to infer the schema
      val headerRow = spark.sparkContext.textFile(file).map(_.split(",", -1)).first()
      val header = Seq(StructField("Id", LongType, false)) ++  headerRow.map(StructField(_, dataType, true, metadata)).toSeq.slice(1 , headerRow.length)

      //return the dataset
      val rdd = spark.sparkContext.textFile(file).filter(!_.startsWith("Id")).map(_.split(",", -1) )
                     .filter(r => samplingFunc(r.head.toLong))
                     .map( r => (r.head.toLong, r.slice(1,r.length).map( s => if(s==""){null}else{cast(s)} ) ) )
      (header, rdd )
  }

  def mergeMap[A](a:Map[A,Int], b:Map[A,Int]):Map[A,Int] = {
     a++b.map{ case (k,v) => k -> (v + a.getOrElse(k,0)) }
  }

  def cellValue(row:Array[Any], col:StructField):Any = { row(col.metadata.getLong("colIndex").toInt) }
  

  def fillMetadataAndSort(header:Seq[StructField]):Seq[StructField] = {
    val format = Pattern.compile("L(\\d+)_S(\\d+)_(.*)");
    header.zipWithIndex.map{ case (c, index) => 

       var stationName:String = ""
       var newName = c.name
       val m = format.matcher(c.name)
       if(m.find()){
          stationName =  "L%01d_S%02d".format(m.group(1).toInt, m.group(2).toInt) 
          newName     =  "%s_%s".format(stationName, m.group(3)) 
       }
       StructField(newName, c.dataType, c.nullable, new MetadataBuilder().withMetadata(c.metadata).putLong("colIndex", index).putString("station", stationName).build )
    }
    .sortWith{ (a,b) => if(a.metadata.getString("station")=="")true else a.name < b.name }
  }

  def getPattern(groupColumns:Seq[Tuple2[String, Seq[StructField]]], row:Array[Any]):String = { 
     groupColumns.map{ g => g._2.map(s => cellValue(row, s)!=null).reduce( (a,b) => a||b) }.map(if(_){"1"}else{"0"}).mkString("")
  }

  def groupByStationBinary(header:Seq[StructField], rdd:RDD[Array[Any]]):Tuple2[Seq[String], RDD[Tuple2[String, Map[String,Int]]]] = {
    val groupColumns = header.filter(_.metadata.getString("station")!="").groupBy(_.metadata.getString("station")).toSeq.sortWith{ (a,b) => a._1 < b ._1 }
    val ResponseColumn = header.filter(_.name=="Response").head

    val binaryRdd = rdd.map{ row =>
//       val key    = groupColumns.map{ g => g._2.map(s => cellValue(row, s)!=null).reduce( (a,b) => a||b) }.map(if(_){1.toInt}else{0.toInt})  //line_station info used as grouping key 
       val key = getPattern(groupColumns, row)
       val value = Map( cellValue(row, ResponseColumn).toString -> 1, "Any" -> 1) //just count 1 per response type and for total
       (key, value)       
    }.reduceByKey{ (a,b) => mergeMap(a,b) }
    .mapValues(m => Map(m.toSeq.sortWith(_._1 < _._1):_*) ) //order items in map 
    .sortBy(_._2("Any"),false) //order patterns by importance

    ( groupColumns.map(_._1) ,   binaryRdd)
  }

  def listPossibleCategories(header:Seq[StructField], rdd:RDD[Array[Any]]):Seq[Tuple2[StructField, Map[String,Int]]] = {
    val catColumns = header.filter(c => c.dataType==StringType && c.metadata.getString("station")!="" && c.metadata.getString("type")=="C")

    val catFreq = rdd.map{ row =>  
       catColumns.map{ col =>
          val v = cellValue(row, col)
          if( v==null){Map[String,Int]()}else{ Map( v.toString -> 1, "Any" -> 1) }      
    }}.reduce{ (ma,mb) => ma.zip(mb).map{case (a,b) => mergeMap(a,b) } } 
    .map(m => Map(m.toSeq.sortWith(_._1 < _._1):_*) ) //order items in map 

    catColumns.zip(catFreq)
  }

  def printStat(spark:SparkSession, header:Seq[StructField], rdd:RDD[Array[Any]]):Unit = {
     val responseIndex = header.filter(_.name=="Response").head.metadata.getLong("colIndex").toInt
     val doubleCol = header.filter(_.dataType==DoubleType)


     val StatAll = rdd
         .map{row => doubleCol.map{c => 
            val v = cellValue(row, c) 
            val r = row(responseIndex)
            val s = new StatCounter()
            if(v !=null ) s.merge(v.asInstanceOf[Double])
            Map( r -> s )  //return a map to save the results per response type 
         }}
         .reduce( (a,b) => a.zip(b).map( p =>  p._1++p._2.map{ case (k,v) => k -> (v.merge(p._1.getOrElse(k,new StatCounter())) ) } ) )

     
     StatAll.zip(doubleCol).foreach{ p=>
        var buffer:String = "| %20s ".format(p._2.name)

        var statTe0 = p._1.getOrElse( 0 ,new StatCounter())  //Stat for test with response 0
        var statTe1 = p._1.getOrElse( 1 ,new StatCounter())  //Stat for test with response 0
        var statTe  = statTe0.merge(statTe1) 
        var statTr  = p._1.getOrElse(-1 ,new StatCounter())  //Stat for test with response 0
        var statTo  = statTe.merge(statTr) 

        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTe0.count, statTe0.mean, statTe0.sampleStdev, statTe0.min, statTe0.max)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTe1.count, statTe1.mean, statTe1.sampleStdev, statTe1.min, statTe1.max)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTe .count, statTe .mean, statTe .sampleStdev, statTe .min, statTe .max)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTr .count, statTr .mean, statTr .sampleStdev, statTr .min, statTr .max)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTo .count, statTo .mean, statTo .sampleStdev, statTo .min, statTo .max)
        println(buffer + "|")
     }

  }




}


