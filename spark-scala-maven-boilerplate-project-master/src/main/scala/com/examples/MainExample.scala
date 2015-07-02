package com.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import breeze.linalg.{ Vector, DenseVector }
import breeze.linalg.Vector._
import java.util.Random
import scala.math.exp
import scala.math.log
import org.apache.log4j.PropertyConfigurator
import java.io.PrintWriter
import java.io.File
import breeze.linalg.SparseVector
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.apache.commons.collections.Buffer
import java.io.BufferedWriter
import java.io.FileWriter
import org.apache.spark.partial.MeanEvaluator
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
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
	//Get rid of first (label) element and add intercept : 
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
	 * Modif en place de cumGradient ! 
	 * @return loss
	 */
	def calculGradEtLossInPlace(x:Vector[Double], y:Double, weights:Vector[Double], cumGradient:Vector[Double]):Double={
	  val margin = -1.0 * weights.dot(x)
	  cumGradient += x * ((1.0/ (1.0 + math.exp(margin))) - y)
	  val loss = 
	  	if(y>0){
	  	  if(margin>700) margin else math.log1p(math.exp(margin))
	  	}
	  	else{
	  	  if(margin>700) 0 else math.log1p(math.exp(margin))-margin
	  	}
	  return loss
	}


	/**
	 * @return : Writes in file arg(7) -> tsv format
	 * #Test
	 * #executants
	 * percentTraining
	 * #Iteration
	 * miniBatchSize
	 * Count_Cache
	 * Pas
	 * TpsMoySGD
	 * wFinal
	 * Score
	 * TpsScore
	 * TpsTot
	 * Loss
	 * #RealNumExec
	 */
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
	   * arg7 = pathTofileToWrite
	   * arg8 = path to conf file
	   * //Not YET : arg9 = dense/sparse to know which 
	   */
		var strToWrite = arg(1)+ "\t" + arg(2) + "\t" + arg(3) + "\t" + arg(4) + "\t" + arg(5) + "\t"
		var sec = System.currentTimeMillis()
		val secStart = System.currentTimeMillis()
		println("Time in millis at the start: "+ sec )
		
		PropertyConfigurator.configure(arg(8))		
		println("On choisit le bon fichier de configuration pour le logger")
		Logger.getLogger("org").setLevel(Level.WARN)
		Logger.getLogger("akka").setLevel(Level.WARN)
		Logger.getLogger("spark").setLevel(Level.WARN)
		
		val pathToFiles = arg(0)		
		val conf = new SparkConf().setAppName("SGD test on Criteo Dataset")
		conf.setMaster("local[*]")
		val sc = new SparkContext(conf)
	    println("Bonne mise en place du SparkContext")

	    sec = System.currentTimeMillis()
		//Getting and Parsing Data
		val percentData = arg(3).toDouble
		val data = sc.textFile(pathToFiles).map(parseLineCriteoTrain_DV)
		
//		//Normalize : X devient (X - moyenne)/ecartType
//		val seqOp = (c:(DenseVector[Double], DenseVector[Double], scala.Long), v:DataPoint) => {
//		  ((c._1 + (v.x :* v.x)).toDenseVector, (c._2 + v.x).toDenseVector, c._3 + 1L)
//		  }
//		val combOp = (c1:(DenseVector[Double], DenseVector[Double], scala.Long),c2:(DenseVector[Double], DenseVector[Double], scala.Long)) => {
//		  ((c1._1+ c2._1).toDenseVector, (c1._2 + c2._2).toDenseVector, c1._3 + c2._3)
//		  }
//		val featureSize = data.first.x.length
//		val (carre, somme, compte) = data.treeAggregate((DenseVector.zeros[Double](featureSize),DenseVector.zeros[Double](featureSize), 0L))(seqOp, combOp, depth=2)
//		
//		val moyenne = somme/compte.toDouble
//		println(moyenne)
//		println()
//		moyenne.foreach(println)
//		readLine("C'est bon pour la moyenne ? > ")
//		
//		val esperanceAuCarre = moyenne :* moyenne
//		val esperanceDesCarres = carre/compte.toDouble
//		val unSurEcartType = (esperanceDesCarres - esperanceAuCarre).map(x => 1/math.sqrt(x))
//		println(unSurEcartType)
//		println()
//		unSurEcartType.foreach(println)
//		readLine("C'est bon pour l ecart type ? > ")
//		//Fin du calcul de moyenne et variance
		
		val moyenne = DenseVector(0.0, 19.68338822751884, 1060.0243277266534, 213.71905094558392, 60.48550877925574, 93288.07342920362, 904.9704152542275, 162.9635080391697, 133.10426410709087, 1023.8104323290413, 39.20541710858735, 271.98656512411253, 26.100660468858873, 653.7891238244023, 377140.9568311002, 558866.4644557686, 495925.48345359316, 517844.74182640255, 654639.8437000095, 596730.9672947901, 527810.2080114672, 537579.2927661728, 396012.4661558111, 462807.3633273086, 552089.6490133194, 493195.7586611018, 510132.5801904891, 629410.997063303, 506642.9792986425, 480890.5903386074, 433187.912550828, 547976.9410041099, 417267.29874648066, 426274.2016359204, 503566.86781733326, 44882.014297320646, 600825.0935555688, 527790.0450599956, 248032.57627932887, 283052.83741375036)
		val unSurEcartType = DenseVector(1.0, 0.013894925080291155, 2.5662055294567383E-4, 2.8339745670567904E-4, 0.011894208476958141, 6.034119443579218E-6, 3.2574885323666996E-4, 0.0015456354767289728, 0.00599255348330285, 4.617591909947344E-4, 0.016152671698484077, 0.0019538264513617575, 0.003631438822274205, 6.777804392986884E-4, 2.857082166073669E-6, 3.5396077571695426E-6, 3.275430613730722E-6, 3.168434992558608E-6, 3.6476741965059343E-6, 2.954374362090143E-6, 3.240049523339526E-6, 9.090096818218723E-6, 5.221330739173902E-6, 3.5458017789602418E-6, 3.440890774475954E-6, 3.3040770266970405E-6, 3.5028807538341923E-6, 8.903436151478599E-6, 3.3535160649341355E-6, 3.2038067848241315E-6, 4.056320861202199E-6, 3.238682897454082E-6, 2.370370899893361E-6, 2.5891628592515156E-6, 3.13978080585371E-6, 1.0881329046544433E-5, 4.332623640005158E-6, 3.3402524223879458E-6, 3.538401566493884E-6, 3.0575447701419923E-6)

		println(data.first)
		//On modifie les donnees : 
		val madata = data.map(point => {
		  DataPoint((point.x-moyenne):*unSurEcartType, point.y)
		})
		println(madata.first)
		
		val splits = data.randomSplit(Array(percentData, 1.0-percentData), seed = arg(1).toLong)
		val points = splits(0).cache()
		val test = splits(1)
		
		println("Bon chargement des données : "  + (System.currentTimeMillis()-sec))
		
		val ITERATIONS = arg(4).toInt
		sec = System.currentTimeMillis()
		val n = points.count()
		val countAndCache = System.currentTimeMillis()- sec
		strToWrite += countAndCache + "\t"
		println("Count and Cache des données : " + countAndCache)
		
		println(points.first())
		
		val nor: Double = 1.0 / n
		val pas = arg(6).toDouble
		
		println("Pas = " + pas)
		strToWrite += pas + "\t"
		
		// Initialize w to a random or zero value
		/* 75% : 
		 * new DenseVector(Array(0.0,3.881703955986748E-4,1.9335332365441627E-5,1.0138141939316937E-5,-1.4502573629561445E-4,-7.566790898207824E-7,-4.82388419435262E-5,-1.9640848465532707E-4,-0.0010870946682300772,2.514586392987897E-5,0.005169099688167095,3.307830682602945E-4,4.344488306671991E-4,-2.5190152580760846E-4,-3.7593616716024554E-8,-3.2040545512755376E-7,-1.4573812058621003E-7,-7.787216278803954E-8,-5.29515075559877E-8,-1.4892550747605822E-7,-1.417940362503122E-7,-4.0122790338387397E-7,-1.6050504602114809E-7,-2.862502675005578E-7,-2.1295958850715176E-8,1.5929298302171992E-7,3.520368256651298E-8,5.951095917389688E-7,1.5372331347809986E-8,-1.1769151481843464E-7,-8.962831290442038E-7,-6.092784795578786E-8,-1.9000811820135775E-7,3.73178998459603E-7,1.1550411749873857E-7,2.5451507051959573E-7,1.6836314926971445E-7,-1.4723843187344443E-7,4.7319254968399345E-8,-2.832392954350416E-7))
		 */
		
		
		var w = DenseVector.fill(D+1){0.0}
		val featureSize = D+1
		
		//To be plotted	
		val lossHistory = new ArrayBuffer[Double](ITERATIONS)
		val timeHistory = new ArrayBuffer[Double](ITERATIONS)
		val wHistory = new ArrayBuffer[breeze.linalg.DenseVector[Double]](ITERATIONS)
		
		val sampleSize = arg(5).toDouble
		
		require(n * sampleSize >= 1, s"Size of sample too small : got $sampleSize for $n training examples" )
		println("miniBatchSize = " + (n*sampleSize))
		
		
		/**
		 * TODO/TOTry: gradient = sc.aggregate ? 
		 */
		for (i <- 1 to ITERATIONS) {
		  println("Debut de l'iteration : " + i)
		  val secDebutIter = System.currentTimeMillis()
		  
		  //Broadcast the weights vector :
		  val bcW = points.context.broadcast(w);
		  
		  /**
		   * @param c : un triplet (gradient:Vector[Double], loss:Double, count:Long)
		   * @param v : un DataPoint
		   * 
		   * @return c 'like' object
		   */
		  val seqOp = (c:(DenseVector[Double],Double, scala.Long),v:DataPoint) => {
			  val loss =  calculGradEtLossInPlace(v.x, v.y, bcW.value, c._1)
			  (c._1, c._2 + loss, c._3 +1)
		  }
		  
		  /**
		   * Merge two c 'like' object (cf. au dessus)
		   */
		  val combOp = (c1:(DenseVector[Double], Double, scala.Long), c2:(DenseVector[Double], Double, scala.Long))=>{
			  ((c1._1+ c2._1).toDenseVector,c1._2 + c2._2, c1._3 + c2._3)
		  }
		  
		  /**
		   * Usage of "sample" method :
		   * points.sample(withReplacement, fraction, seed)
		   */
		  
		  //Depth = 2 is default value. Try other ones ? 
		  val (gradientSum, lossSum, miniBatchSize) = points.sample(false, sampleSize, seed=i.toLong)
			.treeAggregate((DenseVector.zeros[Double](featureSize), 0.0, 0L))(seqOp, combOp, depth=2)
		  
		  val stepSize = -pas /*On ne fait plus la division par racine de i /math.sqrt(i)*/
		  w += (gradientSum/miniBatchSize.toDouble) * stepSize
			
//		  Append Loss And Time of this iteration :
		  // WARNING : Loss is calculated on the data sample chosen ! Not on whole dataset !
		  val loss = lossSum/miniBatchSize.toDouble
		  println("Loss = " + loss)
		  lossHistory.append(loss)
		  timeHistory.append((System.currentTimeMillis()-secDebutIter)/1000.0)
		  wHistory.append(w.toDenseVector)
		  		  
		}
		
		//Adding Temps Moyen Iteration : 
		strToWrite += timeHistory.reduce(_+_)/ITERATIONS.toDouble + "\t"

		//On choisit la moyenne des w pour prédire ! 
		//On écrit le vecteurs dans le fichier
		println("On reduce les w !")
		val wFinal = wHistory.reduce((a,b) => (a+b).toDenseVector)/ITERATIONS.toDouble
		strToWrite += wFinal.toString() + "\t"
		val bcLastW = sc.broadcast(wFinal)
		
		//Prediction sur test :
		println("On prédit !")
		sec = System.currentTimeMillis()
		val (nbBon, tot) =  test.map(p => (if(decision(hypothesis(bcLastW.value, p.x))==p.y) 1 else 0, 1)).reduce((a,b) => (a._1+ b._1, a._2+b._2))
		val score = nbBon.toDouble / tot
		
		//Adding the score and time of scoring !
		strToWrite += score +"\t" + java.lang.String.valueOf(System.currentTimeMillis()-sec)+ "\t"
		
		//Total time of program
		val totalTime = System.currentTimeMillis()-secStart
		strToWrite += totalTime+"\t"
		
		//Write to file
		val writer = new PrintWriter(new BufferedWriter(new FileWriter(arg(7), true)))
		
		strToWrite += lossHistory.mkString("\t")+"\t"
		strToWrite += "TIME\t"+timeHistory.take(if(ITERATIONS<100) ITERATIONS else 100).mkString("\t")
		strToWrite += "\t" + (sc.getExecutorMemoryStatus.toArray.length -1)
		//strToWrite += arg(9)
		
        writer.println(strToWrite)
        writer.close()
        sc.stop()
		
		
	}
}
