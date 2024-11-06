from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

# Loading the data
df = spark.read.csv("powerbi.csv", header=True, inferSchema=True)

label_column = "Churn"
feature_columns = [col for col in df.columns if col != label_column]

df = df.withColumn(label_column, col(label_column).cast("int"))

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# Splitting data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(labelCol=label_column, featuresCol="features", numTrees=100, maxDepth=5, seed=42)
rf_model = rf.fit(train_data)

predictions = rf_model.transform(test_data)

# Evaluating model with AUC metric
evaluator = BinaryClassificationEvaluator(labelCol=label_column, metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")

accuracy = predictions.filter(predictions[label_column] == predictions.prediction).count() / float(predictions.count())
print(f"Accuracy: {accuracy}")

predictions.groupBy(label_column, "prediction").count().show()
