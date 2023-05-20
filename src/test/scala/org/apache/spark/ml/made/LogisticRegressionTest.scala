package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}
import org.apache.spark.sql.functions._

class LogisticRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  private def validateModelMetrics(model: LogisticRegressionModel, data: DataFrame, threshold: Double): Unit = {
    val pred_df = model.transform(data)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val roc = evaluator.evaluate(pred_df)
    roc should be >= threshold
  }
  
  "Model" should "get bert AOC on noisy dataset" in {
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val model = lr.fit(noisy_df)

    val roc_best = 0.9488086284424957
    validateModelMetrics(model, noisy_df, roc_best)
  }
  "Model" should "get best AOC and weights on clear dataset" in {
    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val model = lr.fit(clear_df)

    val roc_best = 1
    validateModelMetrics(model, clear_df, roc_best)

    val weights = model.getWeights()
    val weights_best = Array(0.02300168, 0.1247883, -0.39241986, -0.19301087, -0.60742174)
    val delta = 0.05

    weights(0) should be(weights_best(0) +- delta)
    weights(1) should be(weights_best(1) +- delta)
    weights(2) should be(weights_best(2) +- delta)
    weights(3) should be(weights_best(3) +- delta)
    weights(4) should be(weights_best(4) +- delta)
  }

  "Model" should "correctly work with rawPredictionCol and predictionCol" in {
    val lr = new LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setRawPredictionCol("rawPrediction")
        .setPredictionCol("prediction")
      val model = lr.fit(noisy_df)

      var df1 = model.transform(noisy_df)
      val func = udf((t: Double) => if (t >= lr.getThreshold) 1.0 else 0.0)
      df1 = df1.withColumn("prediction", when(lit("rawPrediction") >= 0.5, 1.0).otherwise(1))
      val sum1 = df1.select(sum("prediction"))
      val df2 = model.predict(noisy_df)
      val sum2 = df2.select(df2("prediction"))

      sum1 == sum2
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setRawPredictionCol("rawPrediction")
        .setPredictionCol("prediction")
    ))

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline
      .load(tmpFolder.getAbsolutePath)
      .fit(noisy_df)
      .stages(0)
      .asInstanceOf[LogisticRegressionModel]

    val roc_best = 0.9488086284424957
    validateModelMetrics(model, noisy_df, roc_best)
  }
}
