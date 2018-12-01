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

