package kaggle

import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer

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
import java.io.PrintWriter

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
//    val samplingFunc:Long=>Boolean = (Id => Id%1000<100) //only process one percent of the dataset
    val samplingFunc:Long=>Boolean = (Id => true) //all rows

    var (header, rdd) = getUnifiedDataset(spark, "file:///home/loicus/Data/Code/Kaggle/Bosch/inputData/"    , samplingFunc )


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
    val stat = getStat(spark, header, rdd)    
    stat.foreach{ p=>
        val statTr0 = p._2.getOrElse( 0 ,new StatCounter())  //Stat for test with response 0
        val statTr1 = p._2.getOrElse( 1 ,new StatCounter())  //Stat for test with response 0
        val statTr  = statTr0.copy().merge(statTr1) 
        val statTe  = p._2.getOrElse(-1 ,new StatCounter())  //Stat for test with response 0
        val statTo  = statTr.copy().merge(statTe) 

        var buffer:String = "| %20s ".format(p._1.name)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTr0.count, statTr0.mean, statTr0.sampleStdev, statTr0.min, statTr0.max)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTr1.count, statTr1.mean, statTr1.sampleStdev, statTr1.min, statTr1.max)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTr .count, statTr .mean, statTr .sampleStdev, statTr .min, statTr .max)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTe .count, statTe .mean, statTe .sampleStdev, statTe .min, statTe .max)
        buffer += "| %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(statTo .count, statTo .mean, statTo .sampleStdev, statTo .min, statTo .max)
        println(buffer + "|")
    }

    //get all distinct signature by groupping all non null data together
    var (headerLines, patterns) = groupByStationBinary(spark, header, rdd)
    println("Number of different patterns = %d".format(patterns.size))
    println("Lines = (%s)".format(headerLines.mkString(",")))
    patterns.foreach(r => println("Pattern (%s) has %s entries".format(r._1, r._2.mkString(";"))) )

    //get all distinct signature by groupping all non null data together
    val timeDiff = getTimeSpreadAtStation(spark, header, rdd)
    timeDiff.foreach(r => println("Timediff at station %15s with %3d time feature : %+8d - %+8.2f - %+8.2f - %8.2f - %8.2f ".format(r._1._1, r._1._2, r._2.count, r._2.mean, r._2.sampleStdev, r._2.min, r._2.max)))

    //check all different types of category that we can get for each categorical feature
    //WARNING this is quite time consumming
    var categories = getPossibleCategories(spark, header, rdd)
    println("category size = %d".format(categories.length))
    categories.sortWith( (a,b) => a._2.size < b._2.size ).foreach(r => println("%s --> %d CATEGORIES: %s".format(r._1.name, r._2.size, r._2.mkString(";") )) )


    //add one more column with the pattern to the rdd (and to the header)
    header = header ++ Seq(StructField("Pattern", StringType, false, new MetadataBuilder().putLong("colIndex", header.size).putString("station", "").build  ))
    var groupFeatures = header.groupBy(_.metadata.getString("station")).toSeq.sortWith{ (a,b) => a._1 < b ._1 }
    rdd = rdd.map{ row => row++Array(getPattern(groupFeatures.filter(_._1!=""), row)) }
    groupFeatures = header.groupBy(_.metadata.getString("station")).toSeq.sortWith{ (a,b) => a._1 < b ._1 }


    //Identify variables that are useful
    val usefulFeatures = groupFeatures.flatMap{ case(group, features) =>
       val toKeep:ArrayBuffer[StructField] = ArrayBuffer() //empty
       if(group==""){ //not related to a station --> keep all features
          toKeep ++= features

       }else{
          //time features  (keep only the first time feature for station where they are all identical)
          val timeF = features.filter{ h => h.metadata.contains("type") && h.metadata.getString("type")=="D"}
          if(timeDiff.filter(t => t._1._1==group && t._2.mean==0 && t._2.sampleStdev==0).size==0){
              toKeep += timeF.head
          }else{
              toKeep ++= timeF
          }

          //categorical (keep only the one where there is more than 2 options
          val catF = features.filter{ h => h.metadata.contains("type") && h.metadata.getString("type")=="C"}
          catF.foreach{ f =>
             if(categories.filter(c => c._1.name==f.name && c._2.size>2).size>0) toKeep += f  //keep only features that have at least two categories
          }

          //numerical (keep them all)
          val numF = features.filter{ h => h.metadata.contains("type") && h.metadata.getString("type")=="N"}
          toKeep ++= numF
       }

       toKeep
    }
    .sortWith{ (a,b) => 
       if((!a.metadata.contains("station") || a.metadata.getString("station")=="") && (!b.metadata.contains("station") || b.metadata.getString("station")=="")){ a.name < b.name 
       }else if(!a.metadata.contains("station") || a.metadata.getString("station")==""){ true
       }else if(!b.metadata.contains("station") || b.metadata.getString("station")==""){ false
       }else{ a.name < b.name }
    }

    usefulFeatures.foreach(h => println("Variable %s is useful".format(h.name)))
    println("Found %d/%d useful features".format(usefulFeatures.size, header.size))





    //encode the data
    val statPerFeature = stat.map(p => (p._1.name, p._2.getOrElse( 0 ,new StatCounter()).copy().merge(p._2.getOrElse( 1 ,new StatCounter())).merge(p._2.getOrElse(-1 ,new StatCounter()))   ) ).toMap
    val catPerFeature  = categories.filter(p=>p._2.size>2).map(p => (p._1.name, p._2.toSeq.map(kv => kv._1).filter(_!="Any")) ).toMap

    val usefulFeaturesEncoded = usefulFeatures.flatMap{ h =>
       if(h.metadata.contains("type") && h.metadata.getString("type")=="C"){  // one hot encoding 
            val cats:Seq[String]  = catPerFeature(h.name)
            cats.map( c => StructField(h.name + "_" + c, FloatType, h.nullable, new MetadataBuilder().withMetadata(h.metadata).putString("category", c).build  ) )
       }else{
            Seq( h )
       }
    }

    .sortWith{ (a,b) => 
       if((!a.metadata.contains("station") || a.metadata.getString("station")=="") && (!b.metadata.contains("station") || b.metadata.getString("station")=="")){ a.name < b.name 
       }else if(!a.metadata.contains("station") || a.metadata.getString("station")==""){ true
       }else if(!b.metadata.contains("station") || b.metadata.getString("station")==""){ false
       }else{ a.name < b.name }
    }


    println("HEADER ENCODED = (%s)".format(usefulFeaturesEncoded.mkString(",")))



    val responseIndex = usefulFeaturesEncoded.filter(_.name=="Response").head.metadata.getLong("colIndex").toInt
    val normRDD = rdd
//    .filter(r => r(responseIndex).asInstanceOf[Float]!= -1)  //only save test
    .map{ row =>   
      val modifiedRow = usefulFeatures.flatMap{ h =>  //Drop column and normalize
         val value = cellValue(row, h)
         if(value!=null && h.metadata.contains("station") && h.metadata.getString("station")!="" && h.metadata.getString("type")=="N"){  // normalize x --> x-mean/sigma
            val stat = statPerFeature(h.name)             
            Seq( (value.asInstanceOf[Float] - stat.mean)/stat.stdev )
         }else if(value!=null && h.metadata.contains("station") && h.metadata.getString("station")!="" && h.metadata.getString("type")=="D"){  // normalize x --> [0,1]
            val stat = statPerFeature(h.name)
            Seq( (value.asInstanceOf[Float] - stat.min)/(stat.max - stat.min) )
         }else if(h.metadata.contains("station") && h.metadata.getString("station")!="" &&  h.metadata.getString("type")=="C"){  // one hot encoding 
            val cats  = catPerFeature(h.name)
            if(value!=null){
               cats.map( c => if(value==c){1.0}else{0.0}   )
            }else{
               cats.map( c => null )
            }
         }else{
            Seq( cellValue(row, h) )
         } 
      }

      modifiedRow.map{ v => //convert row to string
         v match{
         case null   => ""
         case _:Float => "%.4f".format(v)
         case _:Double => "%.4f".format(v)
         case _ => v.toString
         }
      }.mkString(",")
    }

    normRDD.saveAsTextFile("normalized2/test")
    val out = new PrintWriter("normalized2/header.csv");
//    normRDD.collect.foreach(l => out.println(l))
    out.println(usefulFeaturesEncoded.map(_.name).mkString(","))
    out.close()






    println("All Done")
    spark.stop()
  }


  def getDataset(spark:SparkSession, file:String, samplingFunc:Long => Boolean, dataType:DataType, metadata:Metadata):Tuple2[ Seq[StructField], RDD[Tuple2[Long,Array[Any]]] ] = {
      val cast:String=>Any = { dataType match {
         case _:FloatType => (_.toFloat)
         case _           => (_.toString)
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

  def getUnifiedDataset(spark:SparkSession, inputDir:String, samplingFunc:Long => Boolean):Tuple2[Seq[StructField], RDD[Array[Any]] ] = {
    //load preprocess TRAIN dataset or create it
    val (dsTrainCatH, dsTrainCat) = getDataset(spark, inputDir+"train_categorical.csv", samplingFunc, StringType, new MetadataBuilder().putString("type", "C").build )
    val (dsTrainDatH, dsTrainDat) = getDataset(spark, inputDir+"train_date.csv"       , samplingFunc,  FloatType, new MetadataBuilder().putString("type", "D").build )
    val (dsTrainNumH, dsTrainNum) = getDataset(spark, inputDir+"train_numeric.csv"    , samplingFunc,  FloatType, new MetadataBuilder().putString("type", "N").build )

    val dsTrainH = dsTrainCatH ++ dsTrainDatH.slice(1, dsTrainDatH.length) ++ dsTrainNumH.slice(1, dsTrainNumH.length)
    val dsTrain  = dsTrainCat.join(dsTrainDat).mapValues{case(left,right)=>left++right}.join(dsTrainNum).mapValues{case(left,right)=>left++right}

    //load preprocess TEST dataset or create it
    val (_, dsTestCat) = getDataset(spark, inputDir+"test_categorical.csv", samplingFunc, StringType, new MetadataBuilder().putString("type", "C").build  )
    val (_, dsTestDat) = getDataset(spark, inputDir+"test_date.csv"       , samplingFunc,  FloatType, new MetadataBuilder().putString("type", "D").build )
    var (_, dsTestNum) = getDataset(spark, inputDir+"test_numeric.csv"    , samplingFunc,  FloatType, new MetadataBuilder().putString("type", "N").build )
    dsTestNum = dsTestNum.mapValues(r => r++Array((-1).toFloat)) //Add negative response to the test sample

    val dsTest = dsTestCat.join(dsTestDat).mapValues{case(left,right)=>left++right}.join(dsTestNum).mapValues{case(left,right)=>left++right}
    

    (fillMetadataAndSort(dsTrainH), (dsTrain++dsTest).map(r => Array[Any](r._1) ++ r._2))
//    (fillMetadataAndSort(dsTrainH), dsTrain.map(r => Array[Any](r._1) ++ r._2))
//    (fillMetadataAndSort(dsTrainH), dsTest.map(r => Array[Any](r._1) ++ r._2))
  }



  def mergeMap[A](a:Map[A,Int], b:Map[A,Int]):Map[A,Int] = {
     a++b.map{ case (k,v) => k -> (v + a.getOrElse(k,0)) }
  }

  def cellValue(row:Array[Any], col:StructField):Any = { row(col.metadata.getLong("colIndex").toInt) }
  

  def fillMetadataAndSort(header:Seq[StructField]):Seq[StructField] = {
    val format = Pattern.compile("L(\\d+)_S(\\d+)_(.)(\\d+)");
    header.zipWithIndex.map{ case (c, index) => 

       var stationName:String = ""
       var newName = c.name
       val m = format.matcher(c.name)
       if(m.find()){
          stationName =  "L%01d_S%02d".format(m.group(1).toInt, m.group(2).toInt) 
          newName     =  "~%s_%04d_%s".format(stationName, m.group(4).toInt, m.group(3)) 
//          newName     =  "%s_%s%s".format(stationName, m.group(3), m.group(4)) 
       }
       StructField(newName, c.dataType, c.nullable, new MetadataBuilder().withMetadata(c.metadata).putLong("colIndex", index).putString("station", stationName).build )
    }
//    .sortWith{ (a,b) => if(a.metadata.getString("station")=="")true else a.name < b.name }
    .sortWith{ (a,b) => a.name < b.name }
  }

  def getPattern(groupColumns:Seq[Tuple2[String, Seq[StructField]]], row:Array[Any]):String = { 
     groupColumns.map{ g => g._2.map(s => cellValue(row, s)!=null).reduce( (a,b) => a||b) }.map(if(_){"1"}else{"0"}).mkString("")
  }

  def groupByStationBinary(spark:SparkSession, header:Seq[StructField], rdd:RDD[Array[Any]]):Tuple2[Seq[String], Array[Tuple2[String, Map[String,Int]]]] = {
    val groupColumns = header.filter(_.metadata.getString("station")!="").groupBy(_.metadata.getString("station")).toSeq.sortWith{ (a,b) => a._1 < b ._1 }

    try { 
       val toReturn:Array[Tuple2[String, Map[String,Int]]] = spark.sparkContext.objectFile("preprocessedData/patterns").collect()
       ( groupColumns.map(_._1) ,   toReturn)
    }catch{
       case _ : Throwable => {
       println("processing station patterns")

       val ResponseColumn = header.filter(_.name=="Response").head

       val binaryRdd = rdd.map{ row =>
          val key = getPattern(groupColumns, row)
          val value = Map( cellValue(row, ResponseColumn).toString -> 1, "Any" -> 1) //just count 1 per response type and for total
          (key, value)       
       }.reduceByKey{ (a,b) => mergeMap(a,b) }
       .mapValues(m => Map(m.toSeq.sortWith(_._1 < _._1):_*) ) //order items in map 
       .sortBy(_._2("Any"),false) //order patterns by importance

       val toReturn = binaryRdd 
       toReturn.saveAsObjectFile("preprocessedData/patterns")
       ( groupColumns.map(_._1) ,   toReturn.collect())
     }}
  }


  def getTimeSpreadAtStation(spark:SparkSession, header:Seq[StructField], rdd:RDD[Array[Any]]):Seq[((String,Int),StatCounter)] = {
    try {  
       val toReturn: Array[((String,Int),StatCounter)] = spark.sparkContext.objectFile("preprocessedData/timeSpread").collect()
       toReturn
    }catch{
       case _ : Throwable => {
       println("processing time distribution")

       val groupColumns   = header.filter(c => c.metadata.getString("station")!="" && c.metadata.getString("type")=="D").groupBy(_.metadata.getString("station")).toSeq.sortWith{ (a,b) => a._1 < b ._1 }
       val ResponseColumn = header.filter(_.name=="Response").head

       val timeDiffRdd = rdd.map{ row =>
          val diffTimeAtStation = groupColumns.map{ g =>
             val timeList = g._2.map(s => cellValue(row, s)).filter(c => c!=null && c.isInstanceOf[Float]).map(_.asInstanceOf[Float]) 
             if(timeList.size==0) new StatCounter()  
             else                 new StatCounter().merge(timeList.max - timeList.min) 
          }
          diffTimeAtStation
       }.reduce{ (a,b) => a.zip(b).map(p => p._1.merge(p._2)) }
       val toReturn = groupColumns.map( h => (h._1, h._2.size ) ).zip(timeDiffRdd)
       spark.sparkContext.parallelize(toReturn).saveAsObjectFile("preprocessedData/timeSpread")
       toReturn
    }}
  }


  //get all possible values for each categorical variables
  def getPossibleCategories(spark:SparkSession, header:Seq[StructField], rdd:RDD[Array[Any]]):Seq[(StructField, Map[String,Int])] = {
    try {  
       val toReturn: Array[(StructField, Map[String,Int])] = spark.sparkContext.objectFile("preprocessedData/categories").collect()
       toReturn
    }catch{
       case _ : Throwable => {
       println("processing categories")

       val catColumns = header.filter(c => c.dataType==StringType && c.metadata.getString("station")!="" && c.metadata.getString("type")=="C")

       val catFreq = rdd.map{ row =>  
          catColumns.map{ col =>
             val v = cellValue(row, col)
             if( v==null){Map[String,Int]()}else{ Map( v.toString -> 1, "Any" -> 1) }      
       }}.reduce{ (ma,mb) => ma.zip(mb).map{case (a,b) => mergeMap(a,b) } } 
       .map(m => Map(m.toSeq.sortWith(_._1 < _._1):_*) ) //order items in map 

       val toReturn = catColumns.zip(catFreq)
       spark.sparkContext.parallelize(toReturn).saveAsObjectFile("preprocessedData/categories")
       toReturn
     }}
  }

  //get statistics for each numerical variables (and type of response)
  def getStat(spark:SparkSession, header:Seq[StructField], rdd:RDD[Array[Any]]):Seq[(StructField, Map[Any,StatCounter])] = {

    try {  
       val toReturn: Array[(StructField, Map[Any, StatCounter])] = spark.sparkContext.objectFile("preprocessedData/stats").collect()
       toReturn
    }catch{
       case _ : Throwable => {
        println("processing statistics")

        val responseIndex = header.filter(_.name=="Response").head.metadata.getLong("colIndex").toInt
        val doubleCol = header.filter(_.dataType==FloatType)

        val StatAll = rdd
            .map{row => doubleCol.map{c => 
               val v = cellValue(row, c) 
               val r = row(responseIndex)
               val s = new StatCounter()
               if(v !=null ) s.merge(v.asInstanceOf[Float])
               Map( r -> s )  //return a map to save the results per response type 
            }}
            .reduce( (a,b) => a.zip(b).map( p =>  p._1++p._2.map{ case (k,v) => k -> (v.merge(p._1.getOrElse(k,new StatCounter())) ) } ) )

        val toReturn = doubleCol.zip(StatAll)
       spark.sparkContext.parallelize(toReturn).saveAsObjectFile("preprocessedData/stats")
       toReturn
     }}
  }

}


