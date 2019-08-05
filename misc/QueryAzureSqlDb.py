# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Set up an AKV-backed scope - see https://docs.azuredatabricks.net/user-guide/secrets/secret-scopes.html#create-an-azure-key-vault-backed-secret-scope

# COMMAND ----------

# Get secret values from the Secrets scope backed by Azure Key Vault
sqlHostName = dbutils.secrets.get(scope = "pzadb_scope", key = "sqlHostName")
sqlDatabaseName = dbutils.secrets.get(scope = "pzadb_scope", key = "sqlDatabaseName")
sqlUserName = dbutils.secrets.get(scope = "pzadb_scope", key = "sqlUserName")
sqlPassword = dbutils.secrets.get(scope = "pzadb_scope", key = "sqlPassword")

# Other variables
sqlPort = 1433

# COMMAND ----------

# Verify that values retrieved from secret scope are protected

print(sqlPassword)

# COMMAND ----------

# Prepare SQL connection URL for JDBC connection to Azure SQL DB
sqlUrl = "jdbc:sqlserver://{0}:{1};database={2}".format(sqlHostName, sqlPort, sqlDatabaseName)

connectionProperties = {
  "user" : sqlUserName,
  "password" : sqlPassword,
  "driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# COMMAND ----------

# Pushdown query to Azure SQL DB

# Can use a table outright....
# df = spark.read.jdbc(url = sqlUrl, table = "data.dim_trip_type", properties = connectionProperties)

# Or alias a query to a table and use that
# pushdown_query = "(select * from data.dim_vendor) dim_vendor"
pushdown_query = "(select top 10 * from data.fact_trips_all) top_10_trips"
df = spark.read.jdbc(url = sqlUrl, table = pushdown_query, properties = connectionProperties)

display(df)

# COMMAND ----------

# Read in parallel from multiple worker nodes

df = spark.read.jdbc(url = sqlUrl, table = "data.dim_trip_type", column = "trip_type_id", lowerBound = 1, upperBound = 100000, numPartitions = 100, properties = connectionProperties)

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- hard code stuff in explicit SQL query - not good
# MAGIC 
# MAGIC CREATE TABLE jdbc_dim_vendor
# MAGIC USING org.apache.spark.sql.jdbc
# MAGIC OPTIONS (
# MAGIC   url "jdbc:sqlserver://pzadfsql1.database.windows.net:1433;database=pzadfdb",
# MAGIC   dbtable "data.dim_vendor",
# MAGIC   user "IN_THE_CLEAR_IS_BAD",
# MAGIC   password "IN_THE_CLEAR_IS_BAD"
# MAGIC )

# COMMAND ----------

# Create query via string concat so nothing sensitive is hardcoded - better

sql_table_name = "data.fact_trips_all"
remote_table_name = "remote_fact_trips_all"

spark.sql("""
  CREATE TABLE IF NOT EXISTS """ + remote_table_name + """ 
  USING org.apache.spark.sql.jdbc
  OPTIONS (
    url 'jdbc:sqlserver://""" + sqlHostName + """:1433;database=""" + sqlDatabaseName + """',
    dbtable '""" + sql_table_name + """',
    user '""" + sqlUserName + """',
    password '""" + sqlPassword + """'
    )"""
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query from external table in Azure SQL DB via its Hive table name
# MAGIC -- This lets others use SQL against a local table name without worrying about connection details
# MAGIC -- This is NOT a pushdown query. The data will first be retrieved and then projected on.
# MAGIC -- For a very large table, be careful. A pushdown query (see above) may make more sense.
# MAGIC 
# MAGIC SELECT * FROM remote_fact_trips_all LIMIT 10

# COMMAND ----------

