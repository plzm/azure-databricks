# Databricks notebook source
# MAGIC %md # Machine Learning Development Template

# COMMAND ----------

# MAGIC %md ## Setup data access and loading data

# COMMAND ----------

# MAGIC %md #### Mounting Blob to Azure Databricks 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #### Loading Data from SQL DW

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #### Loading Data from Blob 

# COMMAND ----------

taxes2013 = spark.read.format("csv").option("header", "true").load("dbfs:/mnt/demo/csv/2013_soi_zipcode_agi.csv").drop("_c0")


# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Exploratory Data Analysis

# COMMAND ----------




# COMMAND ----------

# MAGIC %md ## Data Cleaning and enrichment

# COMMAND ----------

display(taxes2013.summary())
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
import json
import requests
import numpy as np
import time
APIKEY = "ff15bba527af804811d252aef420c02d"
endpoint= "http://api.openweathermap.org/data/2.5/forecast?zip={0},us&APPID={1}"
taxes2013_l = taxes2013.limit(20)
def get_temp_forecast(zipcode):
      response = requests.post(endpoint.format(zipcode,APIKEY))
      #Dealing with threshold exceeding exception, retry util we can call the api
      while response.status_code == 429:
        time.sleep(1)
        response = requests.post(endpoint.format(zipcode,APIKEY))

      if response.status_code == 200:
          return json.loads(response.content.decode("utf-8"))['list'][0]['main']['temp']

      else:
          return(response.status_code)
#           raise Exception(str(response.status_code)+":" +response.text )
  
get_temp_forecast_udf = udf(get_temp_forecast)  

display(taxes2013_l.withColumn("forecast_temp", get_temp_forecast_udf("zipcode")))



# COMMAND ----------

# MAGIC %md ## Data Preperation for Machine Learning Model 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## ML for Classification

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## ML for Regression

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Timeseries forecast

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Scoring and Persisting Result to SQL DW

# COMMAND ----------

