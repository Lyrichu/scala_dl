package pers.lyrichu.scaladl.activations

import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms

object Activations {

  def stepFunction(x:INDArray):INDArray = {
    x.cond(Conditions.greaterThan(0.0)).castTo(DataType.INT32)
  }

  def sigmoid(x:INDArray):INDArray = {
    Nd4j.ones(x.shape():_*).divi(
      Transforms.exp(x.mul(-1)).add(1)
    ).castTo(DataType.DOUBLE)
  }

  def relu(x:INDArray):INDArray = {
    Transforms.max(x,0)
  }

  def softmax(x:INDArray):INDArray = {
    Transforms.softmax(x)
  }

}
