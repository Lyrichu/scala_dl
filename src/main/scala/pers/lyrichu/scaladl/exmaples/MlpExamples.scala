package pers.lyrichu.scaladl.exmaples

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import pers.lyrichu.scaladl.activations.Activations

import scala.collection.mutable

class MlpExamples {

  def initNetwork():Map[String,INDArray] = {
    val network = mutable.Map[String,INDArray]()

    network.put("W1",Nd4j.create(Array[Array[Double]](
      Array[Double](0.1,0.3,0.5),Array[Double](0.2,0.4,0.6))))
    network.put("b1",Nd4j.create(Array[Double](0.1,0.2,0.3)))

    network.put("W2",Nd4j.create(Array[Array[Double]](
      Array[Double](0.1,0.4),
      Array[Double](0.2,0.5),
      Array[Double](0.3,0.6)
    )))
    network.put("b2",Nd4j.create(Array[Double](0.1,0.2)))

    network.put("W3",Nd4j.create(Array[Array[Double]](
      Array[Double](0.1,0.3),
      Array[Double](0.2,0.4)
    )))
    network.put("b3",Nd4j.create(Array[Double](0.1,0.2)))
    network.toMap
  }

  def forward(network:Map[String,INDArray],x:INDArray):INDArray = {
    val W1 = network("W1")
    val W2 = network("W2")
    val W3 = network("W3")

    val b1 = network("b1")
    val b2 = network("b2")
    val b3 = network("b3")

    val a1 = x.mmul(W1).add(b1)
    val z1 = Activations.sigmoid(a1)
    val a2 = z1.mmul(W2).add(b2)
    val z2 = Activations.sigmoid(a2)
    val a3 = z2.mmul(W3).add(b3)
    a3
  }
}
