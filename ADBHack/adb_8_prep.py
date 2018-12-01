# Databricks notebook source
# MAGIC %md
# MAGIC References
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameNaFunctions<br>

# COMMAND ----------

from pyspark.sql import *

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get a dataframe for notebook tasks

# COMMAND ----------

# MAGIC %run ./adb_3_ingest_to_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cleanup nulls, bad values

# COMMAND ----------

# How many rows in the starting data

df_flights_full.count()

# COMMAND ----------

# How many nulls do we have in a column

df_flights_full.where(df_flights_full["ArrDelay"].isNull()).count()

# COMMAND ----------

# How many non-nulls do we have in a column - alternate syntax

df_flights_full.filter("Carrier is not NULL").count()

# COMMAND ----------

# Drop rows where ALL columns contain null

df1 = df_flights_full.na.drop(how='all')
df1.count()

# COMMAND ----------

# Drop rows where ANY column contains null

df2 = df_flights_full.na.drop(how='any')
df2.count()

# COMMAND ----------

# Fill nulls in an int column with value 0

df3 = df_flights_full.na.fill(0, ["OriginAirportID"])

# COMMAND ----------

# Replace some values
df4 = df3\
  .na.replace("Palau", "PW", ["state"])\
  .na.replace("Federated States of Micronesia", "FSM", ["state"])

df4.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Clean data for further processing - put some of this together

# COMMAND ----------

df_flights_full_clean = df_flights_full\
  .dropDuplicates()\
  .na.fill(0, ["DepDelay", "ArrDelay"])\
  .na.replace("Palau", "PW", ["state"])\
  .na.replace("Federated States of Micronesia", "FSM", ["state"])

# COMMAND ----------

df_flights_full_clean.count()

# COMMAND ----------

df_flights_full.count() - df_flights_full_clean.count()

# COMMAND ----------

group_by_column = "OriginState"
count_column = "n"

# Group by state so we can count airports per state; rename count column (confusing with describe() otherwise); sort by count descending
# We also filter to US states only so we can use the Databricks map visualization
df4_usstates_only = df4\
  .filter("OriginState NOT IN ('DC', 'FSM', 'PR', 'PW', 'TT', 'VI')")\
  .groupBy(group_by_column)\
  .count()\
  .withColumnRenamed("count", count_column)\
  .sort(group_by_column, ascending=True)

# COMMAND ----------

display(df4_usstates_only)

# COMMAND ----------

