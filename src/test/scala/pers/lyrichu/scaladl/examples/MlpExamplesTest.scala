package pers.lyrichu.scaladl.examples

import org.junit.Test
import org.nd4j.linalg.factory.Nd4j
import pers.lyrichu.scaladl.exmaples.MlpExamples

class MlpExamplesTest {

  @Test
  def testForward() = {
    val mlp = new MlpExamples
    val network = mlp.initNetwork()
    val x = Nd4j.create(Array[Double](1.0,0.5)).reshape(1L,2L)
    val y = mlp.forward(network,x)
    println(y)
  }
}
