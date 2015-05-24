package com.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import breeze.linalg.{Vector, DenseVector,SparseVector}
import java.util.Random
import scala.math.exp
import scala.math.log
import org.apache.log4j.Level
import java.io.PrintWriter
import java.io.File
import breeze.linalg.SparseVector
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import java.lang.Long
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionWithLBFGS}
import java.io.PrintWriter
import java.io.FileWriter
import java.io.BufferedWriter

object MainMLLib {
	val D = 39 // Number of dimensions
			val N = 1048576
			val rand = new Random(42)

case class DataPoint(x: Vector[Double], y: Double)

def parsePoint(line: String): DataPoint = {
		val tok = new java.util.StringTokenizer(line,",")
		var x = new Array[Double](D + 1)
		x(0) = 1.0
		var i = 0
		while (i < D) {
			x(i + 1) = tok.nextToken().toDouble;
			i += 1
		}
		var y = tok.nextToken().toDouble
				val ve = new DenseVector(x)
		return DataPoint(ve, y)
	}
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
	 * Parsage pour train.tiny.csv avec des DenseVector
	 */
	def parseLineCriteoCSV_DV(line:String):DataPoint={
		//Assuming that the first line was removed.
		var myArray = line.split(',')
				myArray = completer(myArray, 41)
				val label = myArray(1)
				//Get rid of first (label) and second (Id) element : 
				var myArray2: Array[Double] = ("1"+:myArray.tail.tail).zipWithIndex.map{ x =>
				myHashFunc(x._1, x._2, N)
				}
				return DataPoint(DenseVector(myArray2), label.toDouble)	
	}

	/**
	 * Parsage pour train.tiny.csv avec des SparseVector
	 */
	def parseLineCriteoCSV_SV(line:String):DataPoint={
		//Assuming that the first line was removed.
		var myArray = line.split(',')
				myArray = completer(myArray, 41)
				val label = myArray(1)
				//Get rid of first (label) and second (Id) element : 
				val myArray2: Array[(Int,Double)] = ("1"+:myArray.tail.tail).zipWithIndex
				.filter(x => if (x._2 >0 && x._2 < 14){
					!(x._1.isEmpty) && !(x._1.toInt==0)
				}
				else{
					!(x._1.isEmpty)
				})
				.map{ x => (x._2,myHashFunc(x._1, x._2, N))
				}
				val (indices, values) = myArray2.unzip 
		return DataPoint(new SparseVector(indices.toArray, values.toArray, 40), label.toDouble)	
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
		return DataPoint(DenseVector(myArray2), label.toDouble)	
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

	def parseLineCriteoTrain_LabeledDV(line:String):LabeledPoint={
		//Assuming that the first line was removed.
		var myArray = line.split('\t')
				myArray = completer(myArray, 40)
				val label = myArray(0)
				//Get rid of first (label) element : 
				val myArray2: Array[Double] = ("1"+:myArray.tail).zipWithIndex.map{ x =>
				myHashFunc(x._1, x._2, N)
				}
				return LabeledPoint(label.toDouble, Vectors.dense(myArray2))
	}
	def main(args: Array[String]): Unit = {
		/*
		 * Usage : 
		 * arg0 = Path to dataset
		 * arg1 = Pourcent du dataset en training set
		 * arg2 = "SGD" ou "LBFGS"
		 * arg3 = Nb iterations SGD (ou nb iterations max LBFGS)
		 * arg4 = learning rate (= pas du SGD)
		 * arg5 = Writing file
		 */
		val secTot = System.currentTimeMillis();
		var sec = secTot;
		val pathToFile= args(0)
		val percentData = args(1).toDouble
		val descentType = args(2)
		val numIter = args(3).toInt
		var pas = 0.0
		val N = 1048576
		// String type : percentData\tTaille\tTemps Cache\tTemps SGD\tTaille du pas\tNombre de pas\tBonne pred\tNb pred\tTemps LBFGS\tBonne pred\tNb pred\n
		var strToWrite = percentData + "\t"
		

		val conf = new SparkConf().setAppName("Test on MLLib")
		val sc = new SparkContext(conf)

		Logger.getLogger("org").setLevel(Level.WARN)
		Logger.getLogger("akka").setLevel(Level.WARN)
		Logger.getLogger("spark").setLevel(Level.WARN)

		val data = sc.textFile(pathToFile).map(parseLineCriteoTrain_LabeledDV)
		val splits = data.randomSplit(Array(percentData,1.0-percentData), seed=1L)
		val training = splits(0).cache()
		val test = splits(1)
		
		println("Bon chargement des données")
		
		sec = System.currentTimeMillis()
		val n = training.count()
		val tpsCache = System.currentTimeMillis() - sec
		println("Temps de Cache :" + tpsCache)
		
		strToWrite += n + "\t" + tpsCache + "\t"

		/*
		 * SGD 
		 */
		//if(descentType.equals("SGD")){
		pas = args(4).toDouble
		sec = System.currentTimeMillis()
		val modelLogRegSGD = LogisticRegressionWithSGD.train(training, numIter, pas, 1.0)
		println()
		val tpsSGD = System.currentTimeMillis() - sec
		println("Temps d'exec du GD : " + (tpsSGD))
		println()

		//Prediction : 
		sec = System.currentTimeMillis()
		val goodAndTot = test.map(x => (if(modelLogRegSGD.predict(x.features)==x.label) 1 else 0, 1)).reduce((a,b) => (a._1 + b._1, a._2 + b._2))
		val tpsPredict = System.currentTimeMillis() - sec
		strToWrite += tpsSGD +"\t"
		strToWrite += pas +"\t"
		strToWrite += numIter +"\t"
		strToWrite += goodAndTot._1 +"\t"
		strToWrite += goodAndTot._2 +"\t"
		
		//}
		/*
		 * LBFGS
		 */
		//if(descentType.equals("LBFGS")){
		sec = System.currentTimeMillis()
		val logRegLBFGS = new LogisticRegressionWithLBFGS()
		logRegLBFGS.optimizer.setNumIterations(numIter).setRegParam(0.0).setConvergenceTol(1E-4)
		val modelLBFGS = logRegLBFGS.run(training)
		println()
		val tpsLBFGS = System.currentTimeMillis() - sec
		print("Temps d'exec du LBGFS : " + (tpsLBFGS))
		println()
		strToWrite += tpsLBFGS +"\t"
		
		//Prediction
		sec = System.currentTimeMillis()
		val goodAndTotLBFGS = test.map(x => (if(modelLBFGS.predict(x.features)==x.label) 1 else 0, 1)).reduce((a,b) => (a._1 + b._1, a._2 + b._2))
		val tpsPredictLBFGS = System.currentTimeMillis() - sec
		
		strToWrite += goodAndTotLBFGS._1 +"\t"
		strToWrite += goodAndTotLBFGS._2 +"\t"
		
		//}
		
		//Temps total
		val tpsTot = System.currentTimeMillis()- secTot
		println("Temps tot : " + (tpsTot))
		
		val out = new PrintWriter(new BufferedWriter(new FileWriter(args(5), true)))
		out.println(strToWrite)
		out.close()
		

	}

}