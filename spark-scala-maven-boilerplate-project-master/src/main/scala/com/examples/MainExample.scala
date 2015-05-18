package com.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import breeze.linalg.{ Vector, DenseVector }
import java.util.Random
import scala.math.exp
import scala.math.log
import org.apache.log4j.Level
import java.io.PrintWriter
import java.io.File
import breeze.linalg.SparseVector


object MainExample {
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
		return 0.	
	}
	else if(idx<13){ //L1-13
		return ((str1+idx).toInt % mod).toDouble
	}
	else{ //C1-26
		return (java.lang.Long.parseLong(str1 + idx,16) % mod).toDouble
	}
}

def parseLineCriteoCSV_DV(line:String):DataPoint={
	//Assuming that the first line was removed.
	var myArray = line.split(',')
	val label = myArray(0)
	//Get rid of first (label) and second (Id) element : 
	var myArray2: Array[Double] = myArray.tail.tail.zipWithIndex.map{ x =>
		myHashFunc(x._1, x._2, N)
	}
	return DataPoint(DenseVector(myArray2), label.toDouble)	
}

def parseLineCriteoCSV_SV(line:String):DataPoint={
	//Assuming that the first line was removed.
	var myArray = line.split(',')
	val label = myArray(0)
	//Get rid of first (label) and second (Id) element : 
	val myArray2: Array[(Int,Double)] = myArray.tail.tail.zipWithIndex
		.filter(x => (x._1.isEmpty))
		.map{ x => (x._2,myHashFunc(x._1, x._2, N))
	}
	val (indices, values) = myArray2.unzip 
	return DataPoint(new SparseVector(indices.toArray, values.toArray, 39), label.toDouble)	
}


	def hypothesis(theta: Vector[Double], x: Vector[Double]): Float = {
		val hyp = 1.0 / (1.0 + exp(-theta.dot(x)))
				return hyp.floatValue()
	}

	def decision(h: Float): Int = {
		if (h > 0.5) return 1
				else return 0
	}

	def kro(j: Int, k: Int): Double = {
		if (j == k) return 0.0
				else return 1.0
	}
	
	def ind(fold: List[Int], k:Int): Double = {
		if(fold.contains(k)) return 1.0
		else return 0.0
	}

	def costFunction(theta: Vector[Double], x: Vector[Double], y: Double, nor: Double): Double = {
		val cout = nor * (-y * log(hypothesis(theta, x)) - (1.0 - y) * log(1.0 - hypothesis(theta, x)))
				return cout
	}
	
	


	def main(arg: Array[String]) {
	  
	    var response = "Test n°"+arg(1)+" avec "+arg(2)+" executants."

		var sec = System.currentTimeMillis()
		var secStart = System.currentTimeMillis()
		var line = "Time in millis at the start: "+sec
		println(line)
	    response+="\n"+line
		
	  
		Logger.getLogger("org").setLevel(Level.WARN)
		Logger.getLogger("akka").setLevel(Level.WARN)
		Logger.getLogger("spark").setLevel(Level.WARN)

		var secTemp = System.currentTimeMillis()
		line="On \"éteint\" le logger en "+(secTemp-sec)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp
		
		val pathToFiles = arg(0)//"ex2data1.txt"
		println("Le programme commence")
		

		val conf = new SparkConf().setAppName("test")//.setMaster("local")
		println("Bonne mise en place du SparkConf")
		conf.setMaster("yarn-client")
		println("Bon set du master pour le SparkConf")
		val sc = new SparkContext(conf)
		println("Bonne mise en place du contexte spark")
		
		secTemp = System.currentTimeMillis()
		line="On met spark en place en "+(secTemp-sec)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp

		val data = sc.textFile(pathToFiles)
		//La valeur initiale du cout devrait être de 0,69314
		//La valeur du gradient intial devrait être de (-0,1 ,-12, -11,26)
		//La valeur du theta optimal devrait être (-25.16 , 0.2 , 0.2 )
		println("Bon chargement des données")
		
		secTemp = System.currentTimeMillis()
		line="On charge les données en "+(secTemp-sec)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp

		val points = data.map(parsePoint _).cache()
		
		secTemp = System.currentTimeMillis()
		line="On parse les données en "+(secTemp-sec)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp
		
		val ITERATIONS = 10
		val n = points.count()
		val nor: Double = 1.0 / (n - 1)
		var lips = points.map(p => p.x.dot(p.x)).reduce(_ + _)
		lips = lips * 4 * nor
		val pasIdeal = 1.0 / lips
		
		secTemp = System.currentTimeMillis()
		line="On calcule le pas idéal en "+(secTemp-sec)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp
		
		
		// Initialize w to a random value
		var w = DenseVector.fill(D+1){2.0 * rand.nextDouble - 1.0}
		val pointsWithIndex = points.zipWithIndex //On attribue un indice à chaque point
		
		secTemp = System.currentTimeMillis()
		line="On indice les données en "+(secTemp-sec)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp
		
		val nfolds = 5
		val idx = List.range(0,n)
		util.Random.shuffle(idx)
		
		secTemp = System.currentTimeMillis()
		line="On finit l'initialisation en "+(secTemp-secStart)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp

		
		
		//Ici on commence la boucle qui va permettre de laisser à chaque fois un fold de côté : celui indicé par j
		for (j <- 1 to nfolds) {
			var fold : List[Int]=List()
			for (s<-1 to ((n/nfolds)-1).toInt){
			  
			  var id = idx.apply((((j-1)*n/nfolds)+s).toInt).toInt
			  fold = fold++List[Int](id)
			  
			}
			var numberOfMistakes: Int = 0
			//Ici on commence la boucle qui permet de calculer le classifieur
			
			for (i <- 1 to ITERATIONS) {

				val gradient = pointsWithIndex.map {
				case (p, k) =>
				p.x * ((hypothesis(w, p.x) - p.y) * nor *(1-ind(fold,k.toInt)) )
				}.reduce(_ + _) 
				w -= gradient * pasIdeal

			}
			val indexKey = pointsWithIndex.map { case (k, v) => (v, k) }
			for (s<-1 to ((n/nfolds)-1).toInt){
			val dataToTest = indexKey.lookup(fold.apply(s-1)) 
			if (decision(hypothesis(w, dataToTest(0).x)) != dataToTest(0).y) numberOfMistakes += 1
			}
			
		println("For the "+j+"th fold we have "+numberOfMistakes+" mistakes")	
		}
		
		secTemp = System.currentTimeMillis()
		line="La cross-validation en 5 folds prend "+(secTemp-sec)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp
		

		sc.stop()
		
		secTemp = System.currentTimeMillis()
		line="Le programme prend "+(secTemp-secStart)+" millisecondes"
		println(line)
		response+="\n"+line
		
		val writer = new PrintWriter(new File("test"+arg(1)+"_"+arg(2)+".txt" ))

        writer.write(response)
        writer.close()
		
		
	}
}
