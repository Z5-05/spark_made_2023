package org.apache.spark.ml.made


import breeze.linalg.{axpy, sum, DenseVector => BreezeVector}
import breeze.stats.{mean}
import org.apache.spark.ml.{PredictorParams}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.util._
import org.apache.spark.ml.param.{ParamMap}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder


trait LogisticRegressionParams
  extends HasMaxIter with HasTol with HasThreshold
  with HasFeaturesCol with HasLabelCol with HasPredictionCol with HasRawPredictionCol
  with HasWeightCol with HasStepSize {

  setDefault(maxIter -> 500, tol -> 1E-6, stepSize -> 0.1, threshold -> 0.5)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setRawPredictionCol(value: String): this.type = set(rawPredictionCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(labelCol))) {
      SchemaUtils.checkNumericType(schema, getLabelCol)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getLabelCol))
    }
  }
}

class LogisticRegression(override val uid: String)
  extends Estimator[LogisticRegressionModel]
    with LogisticRegressionParams
    with DefaultParamsWritable
    with MLWritable {

  def this() = this(Identifiable.randomUID("LogisticRegression"))

  override def copy(extra: ParamMap): LogisticRegression = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  protected def init_weights(weights_num: Int): Vector = Vectors.dense(Array.fill(weights_num)(0.0))
  protected def sigmoid(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
  protected def probability(x: BreezeVector[Double], weights: BreezeVector[Double]): Double = sigmoid(sum(x *:* weights))
  protected def gradient(x: BreezeVector[Double], weights: BreezeVector[Double], y: Double): BreezeVector[Double] = x * (probability(x, weights) - y)

  protected def optimize(weights: Vector, grad: Vector): Vector = Vectors.fromBreeze(weights.asBreeze - $(stepSize) * grad.asBreeze)
  
  override def fit(dataset: Dataset[_]): LogisticRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights: Vector = init_weights(numFeatures + 1)
    
    val transformUdf = dataset.sqlContext.udf.register(uid + "_grad",
      (x_no_ones: Vector, y: Double) => {
        val one = BreezeVector(1.0)
        val x = BreezeVector.vertcat(one, x_no_ones.asBreeze.toDenseVector)
        val grad = gradient(x, weights.asBreeze.toDenseVector, y)
        Vectors.fromBreeze(grad)
      })
    for (_ <- 0 to $(maxIter)) {
      val dataset_with_grad = dataset.withColumn("grad", transformUdf(dataset($(featuresCol)), dataset($(labelCol))))
      val Row(Row(grad_mean_arr)) = dataset_with_grad
        .select(Summarizer.metrics("mean").summary(dataset_with_grad("grad")))
        .first()
      val grad_mean: Vector = Vectors.fromBreeze(grad_mean_arr.asInstanceOf[Vector].asBreeze)
      weights = optimize(weights, grad_mean)
    }
    copyValues(new LogisticRegressionModel(weights.toDense)).setParent(this)
  }
}


object LogisticRegression extends DefaultParamsReadable[LogisticRegression]


class LogisticRegressionModel private[made](override val uid: String, val weights: DenseVector)
  extends Model[LogisticRegressionModel]
    with LogisticRegressionParams
    with MLWritable {

  def this(weights: DenseVector) =
    this(Identifiable.randomUID("LogisticRegressionModel"), weights)

  override def copy(extra: ParamMap): LogisticRegressionModel = {
    copyValues(new LogisticRegressionModel(weights))
  }
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  protected def sigmoid(x: Double): Double = 1.0 / (1.0 + math.exp(-x))
  protected def probability(x: Vector, weights: Vector): Double = sigmoid(sum(x.asBreeze *:* weights.asBreeze))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_probs",
    (x_no_ones: Vector) => {
        val one = BreezeVector(1.0)
        val x = BreezeVector.vertcat(one, x_no_ones.asBreeze.toDenseVector)
        val pred_proba = probability(weights, Vectors.fromBreeze(x))
        pred_proba
      })
    dataset.withColumn($(rawPredictionCol), transformUdf(dataset($(featuresCol))))
  }

  def predict(dataset: Dataset[_]): DataFrame = {
    val predictUdf = dataset.sqlContext.udf.register(uid + "_pred",
    (x_no_ones: Vector) => {
        val one = BreezeVector(1.0)
        val x = BreezeVector.vertcat(one, x_no_ones.asBreeze.toDenseVector)
        val pred_proba = probability(weights, Vectors.fromBreeze(x))
        val pred_y = if (pred_proba >= $(threshold)) 1.0 else 0.0
        pred_y
      })
    dataset.withColumn($(predictionCol), predictUdf(dataset($(featuresCol))))
  }
  def getWeights(): BreezeVector[Double] = {
    weights.asBreeze.toDenseVector
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val params = Tuple1(weights.asInstanceOf[Vector])

      sqlContext.createDataFrame(Seq(params)).write.parquet(path + "/weights")
    }
  }
}

object LogisticRegressionModel extends MLReadable[LogisticRegressionModel] {
  override def read: MLReader[LogisticRegressionModel] = new MLReader[LogisticRegressionModel] {
    override def load(path: String): LogisticRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val weigths = sqlContext.read.parquet(path + "/weights")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val (params) = weigths.select(weigths("_1").as[Vector]).first()

      val model = new LogisticRegressionModel(params.toDense)
      metadata.getAndSetParams(model)
      model
    }
  }
}