from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql import Row

spark = SparkSession.builder \
    .appName("Qsr Prediction") \
    .enableHiveSupport() \
    .getOrCreate()

data = spark.table("default.vlad_air_quality")

indexer = StringIndexer(inputCol="substance", outputCol="SubstanceIndex", handleInvalid="keep")
indexerUSED = indexer.fit(data)
data = indexerUSED.transform(data)

va = VectorAssembler(inputCols=["year", "month", "SubstanceIndex"], 
                            outputCol="features",
                            handleInvalid="skip")
data = va.transform(data)

data = data.select("features", col("Qsr").alias("label"))

train_data, test_data = data.randomSplit([0.8, 0.2], seed=1)

lr = LinearRegression(featuresCol='features', labelCol='label')

lr_model = lr.fit(train_data)

test_predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
rmse = evaluator.evaluate(test_predictions)

rmse_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
mae_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="mae")
r2_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="r2")

rmse = rmse_evaluator.evaluate(test_predictions)
mae = mae_evaluator.evaluate(test_predictions)
r2 = r2_evaluator.evaluate(test_predictions)

print(f"Root Mean Squared Error (RMSE) on test data: {rmse}")
print(f"Mean Absolute Error (MAE) on test data: {mae}")
print(f"R-squared (R²) on test data: {r2}")

new_data = spark.createDataFrame([
    (2024, 1, "Формальдегид"),
    (2025, 1, "Формальдегид"),
    (2026, 1, "Формальдегид"),
], ["year", "month", "substance"])

new_data = indexerUSED.transform(new_data)
new_data = va.transform(new_data)
new_data = new_data.select("features")

new_prediction = lr_model.transform(new_data)
new_prediction.show(truncate=False)