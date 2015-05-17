import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import java.lang.Long
import breeze.linalg.{DenseVector, Vector, SparseVector}

// D = 2 ** 20
val D = 1048576

case class DataPoint(x: Vector[Double], y: Double)

def myHashFunc(str1:String, idx:Int, mod:Int):Double={
	if(str1.isEmpty){
		return 0.	
	}
	else if(idx<13){ //L1-13
		return ((str1+idx).toInt % mod).toDouble
	}
	else{ //C1-26
		return (Long.parseLong(str1 + idx,16) % mod).toDouble
	}
}

def parseLineCriteoCSV_DV(line:String):DataPoint={
	//Assuming that the first line was removed.
	var myArray = line.split(',')
	val label = myArray(0)
	//Get rid of first (label) and second (Id) element : 
	var myArray2: Array[Double] = myArray.tail.tail.zipWithIndex.map{ x =>
		myHashFunc(x._1, x._2, D)
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
		.map{ x => (x._2,myHashFunc(x._1, x._2, D))
	}
	val (indices, values) = myArray2.unzip 
	return DataPoint(new SparseVector(indices.toArray, values.toArray, 39), label.toDouble)	
}


