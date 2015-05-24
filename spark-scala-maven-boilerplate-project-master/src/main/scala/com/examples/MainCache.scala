package com.examples

import java.util.Random
import breeze.linalg.DenseVector
import breeze.linalg.SparseVector
import breeze.linalg.Vector
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import java.io.PrintWriter
import java.io.File



object MainCache {
  
  	val D = 39 // Number of dimensions
	val N = 1048576
	val rand = new Random(42)

case class DataPoint(x: Vector[Double], y: Double)
  
  def myHashFunc(str1:String, idx:Int, mod:Int):Double={
		if(str1.isEmpty){
			return 0.0	
		}
		else if(idx==0){
			return 1.0
		}
		else if(idx<14){ //L1-13
			return ((str1+idx).toInt % mod).toDouble
		}
		else{ //C1-26
			var mystr = str1
					if(mystr.length < 8) mystr = mystr.concat("00000000")
					return (java.lang.Long.parseLong(mystr.substring(0,8) + idx,16) % mod).toDouble
		} 
	}
  
  	def completer(myArr:Array[String], myLength:Int):Array[String]={
		var myArray = myArr
				if(myArray.length < myLength){
					val missing:Int = (myLength - myArray.length)
							require(missing >= 1)
							for (i <- 1 to missing){ //On complete avec des valeurs ""
								myArray :+= ""		
							}
				}
		return myArray
	}
  	
  		/**
	 * Parsage pour train.txt avec des DenseVector
	 */
	def parseLineCriteoTrain_DV(line:String):DataPoint={
		//Assuming that the first line was removed.
		var myArray = line.split('\t')
				myArray = completer(myArray, 40)
				val label = myArray(0)
				//Get rid of first (label) and second (Id) element : 
				val myArray2: Array[Double] = ("1"+:myArray.tail).zipWithIndex.map{ x =>
				myHashFunc(x._1, x._2, N)
				}
		val ve = DenseVector(myArray2)
				return DataPoint(ve, label.toDouble)	
	}

	/**
	 * Parsage pour train.txt avec des SparseVector
	 */
	def parseLineCriteoTrain_SV(line:String):DataPoint={
		//Assuming that the first line was removed.
		var myArray = line.split('\t')
				myArray = completer(myArray, 40)
				val label = myArray(0)
				//Get rid of first (label) and second (Id) element : 
				val myArray2: Array[(Int,Double)] = ("1"+:myArray.tail).zipWithIndex
				.filter(x => if (x._2 >0 && x._2 < 14){
					!(x._1.isEmpty) && !(x._1.toInt==0)
				}
				else{
					!(x._1.isEmpty)
				})
				.map{ x => (x._2,myHashFunc(x._1, x._2, N))}
				val (indices, values) = myArray2.unzip 
						return DataPoint(new SparseVector(indices.toArray, values.toArray, 40), label.toDouble)	
	}

  def main(args: Array[String]): Unit = {
    /** Usage :
     *  0: path to files
     *  1: id of the test
     *  2: number of executors
     *  3: portion of the dataset to keep
     */
    
    
    var response = "Test n°"+args(1)+" avec "+args(2)+" executants."
		//		var response = ""


		
		Logger.getLogger("org").setLevel(Level.WARN)
		Logger.getLogger("akka").setLevel(Level.WARN)
		Logger.getLogger("spark").setLevel(Level.WARN)


		val pathToFiles = args(0)// "train.tiny.csv"//"ex2data1.txt"
		println("Le programme commence")


		val conf = new SparkConf().setAppName("test")//.setMaster("local")
		println("Bonne mise en place du SparkConf")
		conf.setMaster("yarn-client")
		println("Bon set du master pour le SparkConf")
		val sc = new SparkContext(conf)
		println("Bonne mise en place du contexte spark")



		var data = sc.textFile(pathToFiles)

		val Array(f1,f2)=data.randomSplit(Array(args(3).toDouble/100.0, 0.99))
	  	data=f1
		
		data=data.zipWithIndex.filter(x => x._2 >= 1).map(x => x._1) 
		println("Bon chargement des données")

		
		
		val points = data.map(parseLineCriteoTrain_DV _).cache()
		var sec = System.currentTimeMillis()
		points.first()
		var secTemp=System.currentTimeMillis()
		response+= "\n La première action avant la mise en cache prend "+(secTemp-sec)+" millisecondes"
		sec = System.currentTimeMillis()
		points.first()
		secTemp=System.currentTimeMillis()
		response+= "\n L'action après la mise en cache prend "+(secTemp-sec)+" millisecondes"
		
		
		val writer = new PrintWriter(new File("test_"+args(3)+"_"+args(1)+"_"+args(2)+".txt" ))
		writer.write(response)
		writer.close()
		
		
  }

}