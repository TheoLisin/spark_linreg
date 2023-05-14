package org.apache.spark.ml.made

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, DoubleParam, IntParam}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Transformer, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.functions.lit


trait LinearRegressionParams extends HasFeaturesCol with HasLabelCol with HasOutputCol {

  val numIterations = new IntParam(this, "numIterations", "number of iterations")
  val learningRate = new DoubleParam(this, "learningRate", "learning rate")
  val batchSize = new IntParam(this, "batchSize", "size of batches")

  def setFeatureCol(value: String) : this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setNumIterations(value: Int): this.type = set(numIterations, value)
  def getNumIterations: Int = $(numIterations)

  def setLearningRate(value: Double): this.type = set(learningRate, value)
  def getLearningRate: Double = $(learningRate)

  def setBatchSize(value: Int): this.type = set(batchSize, value)
  def getBatchSize: Int = $(batchSize)

  setDefault(numIterations -> 1000, learningRate -> 0.01, batchSize -> 32)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, StructField(getOutputCol, DoubleType))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val vEncoder : Encoder[Vector] = ExpressionEncoder()
    implicit val dEncoder : Encoder[Double] = ExpressionEncoder()

    val lrData: Dataset[(Vector, Double)] = dataset.select(
      dataset($(featuresCol)).as[Vector],
      dataset($(labelCol)).as[Double]
    )

    val dim = lrData.first()._1.size
    
    var weights = Vectors.dense(Array.fill(dim)(1.0)).asBreeze
    var bias = 1.0

    for (_ <- 0 until $(numIterations)) {
      val (wGrad, bGrad) = lrData.rdd.mapPartitions((data: Iterator[(Vector, Double)]) => {
        val wGradSum = new MultivariateOnlineSummarizer()
        val bGradSum = new MultivariateOnlineSummarizer()

        data.grouped($(batchSize)).foreach ((batch) => {
          val wGradSumBatch = new MultivariateOnlineSummarizer()
          val bGradSumBatch = new MultivariateOnlineSummarizer()
          batch.foreach( v  => {
            val x = v._1.asBreeze
            val y = v._2
            
            val pred = x dot weights + bias
            val diff = pred - y
            val grad = diff * x
            wGradSumBatch.add(mllib.linalg.Vectors.fromBreeze(grad))
            bGradSumBatch.add(mllib.linalg.Vectors.dense(diff))
          })
          wGradSum.add(wGradSumBatch.mean)
          bGradSum.add(mllib.linalg.Vectors.dense(bGradSumBatch.mean(0)))
        })
        Iterator((wGradSum, bGradSum))
      }).reduce((x, y) => {
        (x._1 merge y._1, x._2 merge y._2)
      })

      val bGradMean = bGrad.mean(0)

      weights -= wGrad.mean.asBreeze.toDenseVector *:* getLearningRate
      bias -= getLearningRate * bGradMean
    }
    
    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(weights),
      bias
    )
    ).setParent(this)

  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                           override val uid: String,
                           val weights: DenseVector,
                           val bias: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(weights: Vector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bWeights = weights.asBreeze

    val transformUdf = dataset.sqlContext.udf.register(uid + "_prediction",
      (x : Vector) => {
        (x.asBreeze dot bWeights) + bias
      })
    dataset.withColumn($(outputCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val vectors = weights.toArray :+ bias
      sqlContext.createDataFrame(Seq(Tuple1(Vectors.dense(vectors)))).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      val params = vectors.select(vectors("_1")
        .as[Vector]).first().asBreeze.toDenseVector

      val weights = Vectors.fromBreeze(params(0 until params.size - 1))
      val bias = params(-1)

      val model = new LinearRegressionModel(weights, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}