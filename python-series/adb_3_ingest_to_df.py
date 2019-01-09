# Databricks notebook source
# MAGIC %md
# MAGIC #### References:<br>
# MAGIC https://docs.azuredatabricks.net/user-guide/importing-data.html<br>
# MAGIC https://docs.azuredatabricks.net/spark/latest/faq/join-two-dataframes-duplicated-column.html<br>
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame<br>

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import broadcast, lit

# COMMAND ----------

source_file_airports = "/mnt/hack/csv/sample/dat202/airports.csv"
source_file_flights_raw = "/mnt/hack/csv/sample/dat202/raw-flight-data.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC #### With header, and schema inference

# COMMAND ----------

df_airports = spark\
    .read\
    .format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load(source_file_airports)

# COMMAND ----------

df_flights_raw = spark\
    .read\
    .format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load(source_file_flights_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC #### With header and explicit schema

# COMMAND ----------

schema_airports = StructType([
  StructField("airport_id", IntegerType(), True),
  StructField("city", StringType(), True),
  StructField("state", StringType(), True),
  StructField("name", StringType(), True)
])

# COMMAND ----------

schema_flights = StructType([
  StructField("DayofMonth", IntegerType(), True),
  StructField("DayOfWeek", IntegerType(), True),
  StructField("Carrier", StringType(), True),
  StructField("OriginAirportID", IntegerType(), True),
  StructField("DestAirportID", IntegerType(), True),
  StructField("DepDelay", IntegerType(), True),
  StructField("ArrDelay", IntegerType(), True)
])

# COMMAND ----------

# Read the data into a dataframe, explicitly specifying the schema we defined above

df_airports = spark\
    .read\
    .format("csv")\
    .option("header", "true")\
    .schema(schema_airports)\
    .load(source_file_airports)

# COMMAND ----------

df_flights_raw = spark\
    .read\
    .format("csv")\
    .option("header", "true")\
    .schema(schema_flights)\
    .load(source_file_flights_raw)

# COMMAND ----------

# df_airports is small (365 rows). For later work with larger data, we'll broadcast this dataframe to all worker nodes, so that later joins to airports do not incur extra network traffic.

broadcast(df_airports)

# COMMAND ----------

# Lazy cache - in Spark SQL it would be an eager cache

df_airports.cache()

# COMMAND ----------

df_airports.count()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Join flights and airports

# COMMAND ----------

df2 = df_flights_raw\
  .join(df_airports, df_flights_raw["OriginAirportID"] == df_airports["airport_id"], "leftouter")\
  .drop("airport_id")\
  .withColumnRenamed("city", "OriginCity")\
  .withColumnRenamed("state", "OriginState")\
  .withColumnRenamed("name", "OriginAirportName")

# COMMAND ----------

df_flights_full = df2\
  .join(df_airports, df2["DestAirportID"] == df_airports["airport_id"], "leftouter")\
  .drop("airport_id")\
  .withColumnRenamed("city", "DestCity")\
  .withColumnRenamed("state", "DestState")\
  .withColumnRenamed("name", "DestAirportName")

# COMMAND ----------

df_flights_full.cache()

# COMMAND ----------

display(df_flights_full.describe())

# COMMAND ----------

display(df_flights_full)

# COMMAND ----------

