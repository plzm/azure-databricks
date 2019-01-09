# Databricks notebook source
# MAGIC %md
# MAGIC #### References
# MAGIC https://docs.azuredatabricks.net/user-guide/visualizations/index.html<br>
# MAGIC 
# MAGIC https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame<br>
# MAGIC 
# MAGIC https://stackoverflow.com/questions/tagged/apache-spark+apache-spark-sql+python<br>
# MAGIC https://stackoverflow.com/questions/tagged/databricks+python

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run data ingest so we can use dataframe here

# COMMAND ----------

# MAGIC %run ./adb_3_ingest_to_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dataframe API for data operations

# COMMAND ----------

# Group by state so we can count airports per state; rename count column (confusing with describe() otherwise); sort by count descending
df_airports_by_state = df_airports\
  .groupBy("state")\
  .count()\
  .withColumnRenamed("count", "n")\
  .sort("n", ascending=False)

# COMMAND ----------

# Basic descriptive statistics

display(df_airports_by_state.describe())

# COMMAND ----------

# Spark execution plan

df_airports_by_state.explain()

# COMMAND ----------

display(df_airports_by_state)

# COMMAND ----------

# We can also use dataframe select, passing it a list of column names, which emits a new dataframe that can be operated on with dataframe API

# COMMAND ----------

display(df_airports\
  .select("*")\
  .sort("name", ascending=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark SQL for data operations

# COMMAND ----------

# To use a DF in explicit SQL queries, register it as a temp view (cluster lifetime scope)

df_airports.createOrReplaceTempView("df_airports")

# COMMAND ----------

# A Spark SQL SELECT query will emit a new dataframe. This is the same query as the dataframe API query above, for example.

display(sql("SELECT state, COUNT(state) AS n FROM df_airports GROUP BY state ORDER BY n DESC").limit(10))