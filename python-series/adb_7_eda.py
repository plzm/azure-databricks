# Databricks notebook source
# MAGIC %md
# MAGIC References
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameStatFunctions<br>
# MAGIC https://docs.azuredatabricks.net/user-guide/visualizations/index.html<br>

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get a dataframe for notebook tasks

# COMMAND ----------

# MAGIC %run ./adb_3_ingest_to_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data exploration

# COMMAND ----------

df_flights_full.printSchema()

# COMMAND ----------

df_flights_full.count()

# COMMAND ----------

# See if there are duplicate rows - if so, this will differ from just count()

df_flights_full.distinct().count()

# COMMAND ----------

# How many duplicates?

df_flights_full.count() - df_flights_full.dropDuplicates().count()

# COMMAND ----------

# How many duplicates and missing values?

df_flights_full.count() - df_flights_full.dropDuplicates().dropna(how="any", subset=["DepDelay", "ArrDelay"]).count()

# COMMAND ----------

# Summary statistics

display(df_flights_full.describe())

# COMMAND ----------

# Descriptive stats may not make sense for all columns in the df, so let's just get desc stats for a subset
display(df_flights_full.describe().select("summary", "DepDelay", "ArrDelay"))

# COMMAND ----------

# Get top rows - head(n) or take(n)

display(df_flights_full.head(5))

# COMMAND ----------

df_flights_full.show(Truncate=False)

# COMMAND ----------

# limit(n), head(n), take(n)

display(df_flights_full.head(7))

# COMMAND ----------

df_flights_full.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stats

# COMMAND ----------

df_flights_full.approxQuantile("ArrDelay", [0.25, 0.5, 0.75], 0.1)

# COMMAND ----------

display(df_flights_full.freqItems(["DestAirportID"]))

# COMMAND ----------

# Check correlation between two fields

df_flights_full.corr("DepDelay", "ArrDelay")

# COMMAND ----------

display(df_flights_full.select("DepDelay", "ArrDelay"))

# COMMAND ----------

