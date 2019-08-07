# Databricks notebook source
# MAGIC %md
# MAGIC References:
# MAGIC 
# MAGIC https://docs.azuredatabricks.net/spark/latest/dataframes-datasets/introduction-to-dataframes-python.html<br>
# MAGIC https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame<br>
# MAGIC https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.coalesce<br>
# MAGIC https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.repartition<br>
# MAGIC 
# MAGIC Coalesce: Returns a new DataFrame that has exactly numPartitions partitions.<br>
# MAGIC 
# MAGIC Repartition: Returns a new DataFrame partitioned by the given partitioning expressions. The resulting DataFrame is hash partitioned.<br>

# COMMAND ----------

# MAGIC %run ./adb_1_functions

# COMMAND ----------

# MAGIC %run ./adb_3_ingest_to_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Partitions and repartitioning

# COMMAND ----------

num_partitions = 16

# COMMAND ----------

print(df_flights_full.rdd.getNumPartitions())

# COMMAND ----------

df2 = df_flights_full.repartition(num_partitions, ["OriginAirportID","DestAirportID"])

# COMMAND ----------

# How many partitions were created due to our partitioning expression

print(df2.rdd.getNumPartitions())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parquet

# COMMAND ----------

parquet_path = "/mnt/hack/parquet/sample/dat202/"

# COMMAND ----------

path_coalesce = parquet_path + "coalesce/"
dbutils.fs.rm(path_coalesce, True)

# COMMAND ----------

# coalesce(numPartitions: Int): DataFrame - Returns a new DataFrame that has exactly numPartitions partitions

df_flights_full\
  .coalesce(num_partitions)\
  .write\
  .parquet(path_coalesce)

# COMMAND ----------

CleanupSparkJobFiles(path_coalesce)

# COMMAND ----------

path_repartition = parquet_path + "repartition/"
dbutils.fs.rm(path_repartition, True)

# COMMAND ----------

# Repartition and write

df_flights_full\
  .repartition(num_partitions, ["OriginAirportID","DestAirportID"])\
  .write\
  .parquet(path_repartition)

# COMMAND ----------

CleanupSparkJobFiles(path_repartition)

# COMMAND ----------

path_repartition_partitionby = parquet_path + "repartition-partitionby/"
dbutils.fs.rm(path_repartition_partitionby, True)

# COMMAND ----------

# Repartition and write. Here, we are partitioning the output (write.partitionBy) which will create a folder per value in the partition field
# https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.repartition

df_flights_full\
  .repartition(num_partitions, "Carrier")\
  .write\
  .partitionBy("Carrier")\
  .parquet(path_repartition_partitionby)

# COMMAND ----------

CleanupSparkJobFiles(path_repartition_partitionby)

# COMMAND ----------

