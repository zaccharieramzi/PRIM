package com.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import breeze.linalg.{ Vector, DenseVector }
import java.util.Random
import scala.math.exp
import scala.math.log
import org.apache.log4j.PropertyConfigurator
import java.io.PrintWriter
import java.io.File
import breeze.linalg.SparseVector
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.rdd.RDDFunctions._
//import com.github.fommil.netlib.{NativeSystemBLAS, NativeRefBLAS}

object MainExample {
	val D = 39 // Number of dimensions
	val N = 1048576
	val rand = new Random(42)

//TODO : Serializable ? 
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

///**
// * Parsage pour train.txt avec des DenseVector
// * 
// * @param line : String = Une ligne du dataset
// * @return LabeledPoint 
// */
//def parseLineCriteoTrain_Labeled(line:String):DataPoint={
//	//Assuming that the first line was removed.
//	var myArray = line.split('\t')
//	myArray = completer(myArray, 40)
//	val label = myArray(0)
//	//Get rid of first (label) element : 
//	val myArray2: Array[Double] = ("1"+:myArray.tail).zipWithIndex.map{ x =>
//		myHashFunc(x._1, x._2, N)
//	}
//	return LabeledPoint(label.toDouble, Vectors.dense(myArray2))
//}



	def hypothesis(theta: Vector[Double], x: Vector[Double]): Float = {
		val hyp = 1.0 / (1.0 + math.exp(-theta.dot(x)))
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

	/*
	 * TODO : Merge in one function to compute grad & loss at once !
	 */
	def costFunction(theta: Vector[Double], x: Vector[Double], y: Double): Double = {
	  if(y>0){
	    math.log1p(math.exp(-theta.dot(x)))
	  }
	  else{
	    math.log1p(math.exp(-theta.dot(x))) + theta.dot(x)
	  }
	}
	
	/*
	 * Modification sur place de cumGradient
	 * @return loss
	 */
	//TODO : Modif en place de cumGradient
	def calculGradEtLoss(x:Vector[Double], y:Double, theta:Vector[Double], cumGradient:Vector[Double]):(Vector[Double], Double)={
	  val margin = -1.0 * theta.dot(x)
	  cumGradient += x * ((1.0/ (1.0 + math.exp(margin))) - y)
	  val loss = 
	  	if(y>0){
	  		math.log1p(math.exp(margin))
	  	}
	  	else{
	  		math.log1p(math.exp(margin))-margin
	  	}
	  return (cumGradient, loss)
	}
	
	/**
	 * @param str : String to write
	 * @param file: path to file (to write in)
	 * 
	 * Writing Format : Tab separated value
	 * Numero_du_test	nb_Executants	pourcentDuDataset	Nb_Itérations	
	 */
	def myWritingFunc(str: String, file:String):Unit={
	  
	}


	def main(arg: Array[String]) {
	  /*
	   * Usage : 
	   * arg0 = Path to dataset file
	   * arg1 = numero du test
	   * arg2 = nb executants
	   * arg3 = pourcent du dataset en training (le reste est utilisé en test)
	   * arg4 = nombre iterations
	   * arg5 = miniBatchSize (entre 0 et 1)
	   * arg6 = pas
	   * 
	   */
		
		var sec = System.currentTimeMillis()
		var secStart = System.currentTimeMillis()
		println("Time in millis at the start: "+sec)
		
		//TO CHANGE !
		PropertyConfigurator.configure("/home/martin/spark-1.2.0/conf")		
		println("On choisit le bon fichier de configuration pour le logger")
		
		
		val pathToFiles = arg(0)
		println("Le programme commence")
		
		val conf = new SparkConf().setAppName("SGD test on Criteo Dataset").setMaster("local[*]")
		val sc = new SparkContext(conf)
	    println("Bonne mise en place du SparkContext")

	    sec = System.currentTimeMillis()
		//Getting and Parsing Data
		val percentData = arg(3).toDouble
		val splits  = sc.textFile(pathToFiles).map(parseLineCriteoTrain_DV).randomSplit(Array(percentData, 1.0-percentData), 1L)
		val points = splits(0).cache()
		val test = splits(1)
		
		//data=data.zipWithIndex.filter(x => x._2 >= 1).map(x => x._1) 
		println("Bon chargement des données : "  + (System.currentTimeMillis()-sec))
		
		val ITERATIONS = arg(4).toInt
		sec = System.currentTimeMillis()
		val n = points.count()
		val countAndCache = System.currentTimeMillis()- sec
		println("Count and Cache des données : " + countAndCache)
		
		val nor: Double = 1.0 / n
		//var lips = points.map(p => p.x.dot(p.x)).reduce(_ + _)
		//lips = lips * 4 * nor
		//val pasIdeal = 1.0 / lips
		val pas = arg(6).toDouble
		
		println("Pas = " + pas)
		
		
		// Initialize w to a random or zero value
		var w = DenseVector.fill(D+1){0.0}
//		val pointsWithIndex = points.zipWithIndex //On attribue un indice à chaque point
//		println(pointsWithIndex.first)
		
		secTemp = System.currentTimeMillis()
		line="On indice les données en "+(secTemp-sec)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp
		
//		val nfolds = 5
//		var idx = List.range(0,n)
//		idx=util.Random.shuffle(idx)
		
		
		secTemp = System.currentTimeMillis()
		line="On finit l'initialisation en "+(secTemp-secStart)+" millisecondes"
		println(line)
		response+="\n"+line
		sec=secTemp

		
		
		//Ici on commence la boucle qui va permettre de laisser à chaque fois un fold de côté : celui indicé par j
//		for (j <- 1 to nfolds) {
//			var fold : List[Int]=List()
//			for (s<-1 to ((n/nfolds)-1).toInt){
//			  
//			  var id = idx.apply((((j-1)*n/nfolds)+s).toInt).toInt
//			  fold = fold++List[Int](id)
//			  
//			}
			
		val lossHistory = new ArrayBuffer[Double](ITERATIONS)
		val timeHistory = new ArrayBuffer[Double](ITERATIONS)
		
		//var numberOfMistakes: Int = 0
		//Ici on commence la boucle qui permet de calculer le classifieur
		
		val sampleSize = arg(5).toDouble
		require(n * sampleSize >= 1, s"Size of sample too small : got $sampleSize for $n training examples" )
		
		
		for (i <- 1 to ITERATIONS) {
		  var secDebutIter = System.currentTimeMillis()
		  //Broadcast the weights vector :
		  val bcW = points.context.broadcast(w);
		  
		  /**
		   * @param c : un triplet (gradient:Vector[Double], loss:Double, count:Long)
		   * @param v : un DataPoint
		   * 
		   * @return c 'like' object
		   */
		   
		  val seqOp = (c:(DenseVector[Double],Double, Long),v:DataPoint) => {
			  val (newGrad, loss) =  calculGradEtLoss(v.x, v.y, bcW.value, c._1)
			  (newGrad.toDenseVector, c._2 + loss, c._3 +1)
		  }
		  
		  /**
		   * Merge two c 'like' object (cf. au dessus)
		   */
		  val combOp = (c1:(DenseVector[Double], Double, Long), c2:(DenseVector[Double], Double, Long))=>{
			  (c1._1+ c2._1, c1._2 + c2._2, c1._3 + c2._3)
		  }
		  
		  /**
		   * Usage of "sample" method
		   * points.sample(withReplacement, fraction, seed)
		   */
		  
		  //Depth = 2 is default value. Try other ones ? 
		  val (gradientSum, lossSum, miniBatchSize) = points.sample(false, sampleSize, seed=i.toLong)
			.treeAggregate(DenseVector.zeros[Double](bcW.value.size), 0.0, 0L)(seqOp, combOp, depth=2)
			
//			.map { p =>
//			p.x * ((hypothesis(w, p.x) - p.y) * n)
//			}.reduce(_ + _) 
		  lossHistory.append(lossSum/miniBatchSize)
		  val stepSize = -pas/math.sqrt(i)
		   w += (gradientSum/miniBatchSize.toDouble) * stepSize
			

		   timeHistory.append((System.currentTimeMillis()-secDebutIter)/1000.0)
		}
//		val indexKey = points.map { case (k, v) => (v, k) }
//		for (s<-1 to ((n/nfolds)-1).toInt){
//			val dataToTest = indexKey.lookup(fold.apply(s-1))
//					if (decision(hypothesis(w, dataToTest(0).x)) != dataToTest(0).y) numberOfMistakes += 1
//		}
			
//		println("For the "+ j +"th fold we have "+numberOfMistakes+" mistakes")	
		//}
		
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
