# Databricks notebook source
# MAGIC %md
# MAGIC #### References
# MAGIC 
# MAGIC https://docs.azuredatabricks.net/spark/latest/spark-sql/index.html<br>
# MAGIC http://spark.apache.org/docs/latest/sql-programming-guide.html<br>

# COMMAND ----------

parquet_path = "/mnt/hack/parquet/sample/dat202/repartition/"
table_name = "flights_full"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare database

# COMMAND ----------

display(spark.catalog.listDatabases())

# COMMAND ----------

db_name = "hackdb"

# COMMAND ----------

spark.sql("DROP DATABASE IF EXISTS " + db_name + " CASCADE")

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS " + db_name)

# COMMAND ----------

spark.catalog.setCurrentDatabase(db_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create external table on Parquet files, then query it

# COMMAND ----------

# If this is run without specifying a db name, it will use the current database set above

spark.catalog.listTables(db_name)

# COMMAND ----------

# Another way to list tables

display(sqlContext.tables())

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS " + table_name)

# COMMAND ----------

# String concat

spark.sql("""
CREATE TABLE IF NOT EXISTS """ + table_name + """
(
  DayOfMonth INT,
  DayOfWeek INT,
  Carrier STRING,
  OriginAirportID INT,
  DestAirportID INT,
  DepDelay INT,
  ArrDelay INT,
  OriginCity STRING,
  OriginState STRING,
  OriginAirportName STRING,
  DestCity STRING,
  DestState STRING,
  DestAirportName STRING
)
USING parquet
LOCATION '""" + parquet_path + """'"""
)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Explicit SQL = hard-coding...
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS airports
# MAGIC (
# MAGIC   DayOfMonth INT,
# MAGIC   DayOfWeek INT,
# MAGIC   Carrier STRING,
# MAGIC   OriginAirportID INT,
# MAGIC   DestAirportID INT,
# MAGIC   DepDelay INT,
# MAGIC   ArrDelay INT,
# MAGIC   OriginCity STRING,
# MAGIC   OriginState STRING,
# MAGIC   OriginAirportName STRING,
# MAGIC   DestCity STRING,
# MAGIC   DestState STRING,
# MAGIC   DestAirportName STRING
# MAGIC )
# MAGIC USING parquet
# MAGIC LOCATION "/mnt/hack/parquet/sample/dat202/repartition/"

# COMMAND ----------

# Alternative to spark.catalog.listTables()

display(spark.sql("SHOW TABLES"))

# COMMAND ----------

# MAGIC %sql SHOW TABLES

# COMMAND ----------

display(spark.sql("SELECT * FROM " + table_name + " LIMIT 100"))

# COMMAND ----------

