from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, concat_ws
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Initialize Spark
spark = SparkSession.builder.appName("FakeNewsClassifier").getOrCreate()

# === Task 1: Load & Basic Exploration ===
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("news_data")
df.limit(5).toPandas().to_csv("task1_output.csv", index=False)

# === Task 2: Text Preprocessing ===
df_cleaned = df.withColumn("text", lower(concat_ws(" ", "title", "text")))
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized = tokenizer.transform(df_cleaned)
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned = remover.transform(tokenized)
cleaned.select("id", "title", "filtered_words", "label") \
    .toPandas().to_csv("task2_output.csv", index=False)

# === Task 3: Feature Extraction ===
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
featurized = hashingTF.transform(cleaned)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurized)
rescaled = idfModel.transform(featurized)
indexer = StringIndexer(inputCol="label", outputCol="label_index")
final_df = indexer.fit(rescaled).transform(rescaled)
final_df.select("id", "filtered_words", "features", "label_index") \
    .toPandas().to_csv("task3_output.csv", index=False)

# === Task 4: Model Training ===
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)
lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_data)
predictions = model.transform(test_data)
predictions.select("id", "title", "label_index", "prediction") \
    .toPandas().to_csv("task4_output.csv", index=False)

# === Task 5: Evaluate the Model ===
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")
accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)
pd.DataFrame({"Metric": ["Accuracy", "F1 Score"], "Value": [accuracy, f1_score]}) \
    .to_csv("task5_output.csv", index=False)

# Done
print("✅ All tasks completed and outputs saved as taskX_output.csv")
