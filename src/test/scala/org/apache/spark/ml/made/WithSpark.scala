package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.made.WithSpark._sqlc
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

trait WithSpark {
   lazy val spark = WithSpark._spark
   lazy val sqlc = WithSpark._sqlc

  def create_dataset(path: String): DataFrame = {
      lazy val schema = new StructType()
      .add("X1",DoubleType,false)
      .add("X2",DoubleType,false)
      .add("X3",DoubleType,false)
      .add("X4",DoubleType,false)
      .add("label",DoubleType,false)

      lazy val dataframe = _sqlc.read.format("csv")
      .option("header", "true")
      .schema(schema)
      .load(path)

      lazy val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(Array("X1", "X2", "X3", "X4"))
    .setOutputCol("features")

    lazy val df: DataFrame = assembler
    .transform(dataframe)
    .drop("X1", "X2", "X3", "X4")

    df
  }

  lazy val noisy_df = create_dataset("/home/ilya/MADE/spark_made_2023/src/test/data/dataset_noisy.csv")
  lazy val clear_df = create_dataset("/home/ilya/MADE/spark_made_2023/src/test/data/dataset_clear.csv")
}

object WithSpark {
  lazy val _spark = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc = _spark.sqlContext
}
