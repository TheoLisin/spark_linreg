package org.apache.spark.ml.made

import scala.collection.JavaConverters._
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession, Row}
import breeze.linalg._
import breeze.numerics._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 1e-6
  lazy val data: DataFrame = LinearRegressionTest._data

  "Model" should "make predictions" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.dense(0.5, 1.5, 3.0).toDense,
      bias = 4.0
    ).setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("predictions")

    validateModel(model, model.transform(data))
  }


  "Estimator" should "calculate weights and bias" in {
    val estimator = new LinearRegression()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("predictions")

    estimator.setLearningRate(0.01)
    estimator.setNumIterations(1000)

    val model = estimator.fit(data)

    validateModel(model, model.transform(data))
  }

  "Estimator" should "not learn with zero iterations" in {
    val estimator = new LinearRegression()
      .setFeatureCol("features")
      .setLabelCol("label")
      .setOutputCol("predictions")

    estimator.setNumIterations(0)

    val model = estimator.fit(data)

    model.weights(0) should be(0.0 +- delta)
    model.weights(1) should be(0.0 +- delta)
    model.weights(2) should be(0.0 +- delta)
    model.bias should be(0.0 +- delta)
  }

  "Estimator" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeatureCol("features")
        .setLabelCol("label")
        .setOutputCol("predictions")
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModel(model, model.transform(data))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeatureCol("features")
        .setLabelCol("label")
        .setOutputCol("predictions")
    ))

    val model = pipeline.fit(data)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)
    validateModel(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data))
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame) = {
    val predictions: Array[Double] = data.collect().map(_.getAs[Double](2))
    val label: Array[Double] = data.collect().map(_.getAs[Double](0))

    model.weights.toArray.zip(Vectors.fromBreeze(LinearRegressionTest.trueW).toArray).foreach {
      case (w: Double, tw: Double) => w should be(tw +- delta)
    }

    model.bias should be(LinearRegressionTest.trueB +- delta)

    predictions.length should be(data.count())

    (predictions zip label).foreach {case (p: Double, l: Double) => p should be(l +- delta)}
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val schema: StructType = StructType(
    Array(
      StructField("label", DoubleType),
      StructField("features", new VectorUDT())
    ))

  lazy val bMatrix = DenseMatrix.rand(1000, 3)
  lazy val trueW = DenseVector(0.5, 1.5, 3.0)
  lazy val trueB = 4.0
  lazy val label = bMatrix * trueW + trueB

  lazy val rowData = (0 until label.length).map(i => Row(label(i), Vectors.dense(bMatrix(i, ::).t.toArray)))

  lazy val _data: DataFrame = sqlc.createDataFrame(rowData.asJava, schema)
}