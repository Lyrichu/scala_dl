package pers.lyrichu.scaladl.activations

import org.junit.Test
import org.junit.Assert._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex

class ActivationsTest {

  @Test
  def testStepFunction(): Unit = {
    val x = Nd4j.create(Array[Double](1,-1,3,-2)).reshape(Array[Int](2,2))
    val y = Activations.stepFunction(x)
    assertArrayEquals(x.shape(),y.shape())
    assertEquals(y.get(NDArrayIndex.point(0),NDArrayIndex.point(0)).getInt(0),1)
    assertEquals(y.get(NDArrayIndex.point(0),NDArrayIndex.point(1)).getInt(0),0)
  }

  @Test
  def testSigmoid():Unit = {
    val x = Nd4j.linspace(1,10,8).reshape(2,4)
    val y = Activations.sigmoid(x)
    val numSigmoid = (x:Double) => {
      1.0 / (1 + math.exp(-x))
    }
    for (i <- 0 until 2) {
      for (j <- 0 until 2) {
        assertEquals(y.getDouble(i.toLong,j.toLong),numSigmoid(x.getDouble(i.toLong,j.toLong)),0.001)
      }
    }
  }

  @Test
  def testRelu():Unit = {
    val x = Nd4j.randn(3,3)
    val y = Activations.relu(x)

    val numRelu = (x:Double) => {
      if (x > 0) x else 0.0
    }
    for (i <- 0 until 3) {
      for (j <- 0 until 3) {
        assertEquals(y.getDouble(i.toLong,j.toLong),
          numRelu(x.getDouble(i.toLong,j.toLong)),0.0001)
      }
    }
  }

  @Test
  def testSoftmax():Unit = {
    val x = Nd4j.create(Array[Double](0.3,2.9,4.0))
    val y = Activations.softmax(x)
    println(y)

    // multi dimensions
    val x1 = Nd4j.create(Array[Array[Double]](
      Array[Double](0.3,2.9,4.0),
      Array[Double](0.3,2.9,4.0)
    ))
    val y1 = Activations.softmax(x1)
    println(y1)
  }

}
