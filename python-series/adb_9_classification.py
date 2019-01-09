# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# MAGIC %run ./adb_3_ingest_to_df

# COMMAND ----------

df_flights_full_clean = df_flights_full\
  .dropDuplicates()\
  .na.fill(0, ["DepDelay", "ArrDelay"])

df_flights_full_clean.count()

# COMMAND ----------

# Normalize



# COMMAND ----------

# Feature Engineering



# COMMAND ----------

# Get the numeric columns

df_numerics = df_flights_full_clean.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 15).cast("Int").alias("Late")))

display(df_numerics)

# COMMAND ----------

# Split the data into training and testing sets

splits = df_numerics.randomSplit([0.7, 0.3])
df_train = splits[0]
df_test = splits[1]
train_rows = str(df_train.count())
test_rows = str(df_test.count())
print("Training Rows:" + train_rows + " Testing Rows:" + test_rows)

# COMMAND ----------

# Assemble a feature vector, and designate the label column

assembler = VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features")

training = assembler.transform(df_train).select(col("features"), col("Late").alias("label"))

display(training)

# COMMAND ----------

# Train a model using logistic regression

lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.3)
model = lr.fit(training)

# COMMAND ----------

# Test model with testing data

testing = assembler.transform(df_test).select(col("features"), col("Late").alias("realLabel"))
display(testing)

# COMMAND ----------

# Test the model

prediction = model.transform(testing)
predicted = prediction.select("features", "probability", "prediction", "realLabel")
display(predicted.take(1000))

# COMMAND ----------

