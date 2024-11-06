from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()

# Load the data
df = spark.read.csv("powerbi.csv", header=True, inferSchema=True)

# Define label and feature columns
label_column = "Churn"
feature_columns = [col for col in df.columns if col != label_column]

# Cast label to integer if necessary
df = df.withColumn(label_column, col(label_column).cast("int"))

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)

# Split data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Initialize and train Random Forest classifier
rf = RandomForestClassifier(labelCol=label_column, featuresCol="features", numTrees=100, maxDepth=5, seed=42)
rf_model = rf.fit(train_data)

# Make predictions
predictions = rf_model.transform(test_data)

# Evaluate model with AUC metric
evaluator = BinaryClassificationEvaluator(labelCol=label_column, metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc}")

# Calculate accuracy
accuracy = predictions.filter(predictions[label_column] == predictions.prediction).count() / float(predictions.count())
print(f"Accuracy: {accuracy}")

# Show confusion matrix
predictions.groupBy(label_column, "prediction").count().show()
