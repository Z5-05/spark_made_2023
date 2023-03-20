package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.{col, lit, rand}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta: Double = 0.01
  val weights: Vector = LinearRegressionTest._weights
  val bias: Double = LinearRegressionTest._bias
  val data: DataFrame = LinearRegressionTest._data
  val stepSize: Double = 0.8
  val maxIter: Int = 200


  "Model" should "be trained on data" in {
    validateWeights(data, weights, bias, stepSize, maxIter)
  }

  "Model" should "predict right labels" in {
    validateResults(data, weights, bias, stepSize, maxIter)
  }

  "Model" should "work after re-read" in {
    validateWriteRead(data, weights, bias, stepSize, maxIter)
  }

  private def validateWeights(data: DataFrame
                            , weights: Vector
                            , bias: Double
                            , stepSize: Double
                            , maxIter: Int): Unit = {
    // get random outputs with true_model - with random weights
    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setFeaturesCol("features")
      .setPredictionCol("prediction")
    // get true answers for random model
    val data_with_labels = true_model
      .transform(data)
      .select(col("features"), col("prediction"))
    
    // target pipe
    val untrained_model: LinearRegression = new LinearRegression()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setStepSize(stepSize)
        .setMaxIter(maxIter)
    val model = untrained_model.fit(data_with_labels)

    model.weights(0) should be(weights(0) +- delta)
    model.weights(1) should be(weights(1) +- delta)
    model.weights(2) should be(weights(2) +- delta)
    model.bias should be(bias +- delta)
    }

  private def validateResults(data: DataFrame
                            , weights: Vector
                            , bias: Double
                            , stepSize: Double
                            , maxIter: Int): Unit = {
    // get random outputs with true_model - with random weights
    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setFeaturesCol("features")
      .setPredictionCol("prediction")
    // get true answers for random model
    val data_with_labels = true_model
      .transform(data)
      .select(col("features"), col("prediction"))
    
    // target pipe
    val untrained_model: LinearRegression = new LinearRegression()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setStepSize(stepSize)
        .setMaxIter(maxIter)
    val model = untrained_model.fit(data_with_labels)

    val true_y = true_model.transform(data).collect().map(_.getAs[Double](1))
    val pred_y = model.transform(data).collect().map(_.getAs[Double](1))

    for (i <- pred_y.indices) {
      pred_y(i) should be (true_y(i) +- delta)
    }
  }

  private def validateWriteRead(data: DataFrame
                              , weights: Vector
                              , bias: Double
                              , stepSize: Double
                              , maxIter: Int): Unit = {
    // get random outputs with true_model - with random weights
    val true_model: LinearRegressionModel = new LinearRegressionModel(
      weights = weights.toDense,
      bias = bias
    ).setFeaturesCol("features")
      .setPredictionCol("prediction")
    // get true answers for random model
    val data_with_labels = true_model
      .transform(data)
      .select(col("features"), col("prediction"))
    // get pipeline like in example
    var pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setStepSize(stepSize)
        .setMaxIter(maxIter)
    ))
    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    pipeline = Pipeline.load(tmpFolder.getAbsolutePath)
    pipeline.getStages(0).asInstanceOf[LinearRegression].getStepSize should be(stepSize)
    pipeline.getStages(0).asInstanceOf[LinearRegression].getMaxIter should be(maxIter)
    val model = pipeline.fit(data_with_labels).stages(0).asInstanceOf[LinearRegressionModel]

    model.getFeaturesCol should be("features")
    model.getPredictionCol should be("prediction")
    model.weights(0) should be(weights(0) +- delta)
    model.weights(1) should be(weights(1) +- delta)
    model.weights(2) should be(weights(2) +- delta)
    model.bias should be(bias +- delta)
  }
}


object LinearRegressionTest extends WithSpark {
  import sqlc.implicits._

  lazy val _weights: Vector = Vectors.dense(4.2, 1.4, -0.8)
  lazy val _bias: Double = 2

  // generate data like in example, but from seq of vectors with size 3
  lazy val _vectors = Seq.fill(10000)(Vectors.fromBreeze(DenseVector.rand(3)))
  lazy val _data: DataFrame = {
    _vectors.map(x=>Tuple1(x)).toDF("features")
  }
}