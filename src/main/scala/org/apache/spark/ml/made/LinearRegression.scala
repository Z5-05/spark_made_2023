package org.apache.spark.ml.made

import breeze.linalg.{sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol, HasMaxIter, HasStepSize}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}


trait LinearRegressionParameters extends HasPredictionCol
  with HasFeaturesCol
  with HasStepSize
  with HasMaxIter{
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setStepSize(value: Double): this.type = set(stepSize, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  protected def validateTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkNumericType(schema, getPredictionCol)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression (override val uid: String) 
extends Estimator[LinearRegressionModel]
  with LinearRegressionParameters 
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val datasetExt: Dataset[_] = dataset.withColumn("ones", lit(1))
    val vectorAssembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array($(featuresCol), "ones", $(predictionCol)))
      .setOutputCol("features_ext")

    val vectors: Dataset[Vector] = vectorAssembler
      .transform(datasetExt)
      .select("features_ext")
      .as[Vector]

    val numFeatures: Int = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights: BreezeDenseVector[Double] = BreezeDenseVector.rand[Double](numFeatures + 1)

    for (_ <- 0 until $(maxIter)) {
      val summary = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until weights.size).toDenseVector
          val y = v.asBreeze(weights.size)
          val loss = sum(X *:* weights) - y
          val grad = X * loss
          summarizer.add(fromBreeze(grad))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      weights = weights - $(stepSize) * summary.mean.asBreeze
    }
    // Copies param values from this instance to another instance for params shared by them.
    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(weights(0 until weights.size - 1)).toDense,
      weights(weights.size - 1)
    )).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = validateTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](override val uid: String, val weights: DenseVector, val bias: Double)
  extends Model[LinearRegressionModel] with LinearRegressionParameters with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = {
    copyValues(new LinearRegressionModel(weights, bias), extra)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // In order to use function on Spark SQL, you need to register the function with Spark using spark.udf.register()
    val transformUdf = dataset.sqlContext.udf.register(
      uid + "_transform",
      (x: Vector) => {
        sum(x.asBreeze *:* weights.asBreeze) + bias
      }
    )
    // Returns a new Dataset by adding a column or replacing the existing column that has the same name
    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Vector) = weights.asInstanceOf[Vector] -> Vectors.fromBreeze(BreezeDenseVector(bias))

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val (weights, bias) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Vector]).first()

      val model = new LinearRegressionModel(weights.toDense, bias(0))
      metadata.getAndSetParams(model)
      model
    }
  }
}