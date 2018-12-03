# Databricks notebook source
# MAGIC %md # Machine Learning Development Template

# COMMAND ----------

# MAGIC %md ### Creating fake data (optional)

# COMMAND ----------

data_case3#fake data generation for case 3
from faker import Faker
fake = Faker()
fake.latitude()

#List of fields from customer
# ACCOUNT_NO - unique customer account number
# REVENUE_MONTH_DT - billing revenue month
# ACCOUNT_TYP_CD - Residential, Commercial, or Industrial.  (Only Residential accounts were used.)
# ACCOUNT_CREATE_DT - Service start date
# ACCOUNT_STATUS_CD - Active, Final Billed, Pending, etc
# PREMISE_NO - unique premise number
# CUSTOMER_NO - unique customer number
# ACCOUNT_CREDIT_GRP_CD - Credit Action Exempt, Arrears, Balance, etc
# ACCOUNT_EPP_IN - Is account currently on Budget Billing (Y/N)
# LOCAL_OFFICE_NO - geographic account local office
# UTILITY_GROUP_CD - code indicating electric, gas, and unmetered services at the account
# ZIP_CD - postal zip code of the premises
# ELECTRIC_BILLED_KWH_QT - electric kilowatts per hour billed
# ELECTRIC_BILLED_AM - dollar amount for electric kilowatts billed
# GAS_BILLED_THERMS_QT - gas therms billed
# GAS_BILLED_AM - dollar amount for gas therms billed
# UNMETERED_BILLED_AM - dollar amount for unmetered services
# TOTAL_BILLED_AM - total dollar amount billed
import pandas as pd
import numpy as np
import random
from dateutil import parser
import datetime
from dateutil.relativedelta import relativedelta
import datetime
max_num_range = 100
#pattern of consumption of eletricity
ele_pattern = [6,7,6,5,6,7,8,9,8,6,5,6]
#pattern of consumption of gas
gas_pattern = [10,10,8,7,6,5,5,6,6,7,8,9]

def bill_gen(total_mean, pattern):
  month_mean = total_mean/sum(pattern)
  return [month_mean*(1+np.random.randint(-1,1)/2)*month for month in pattern]


def date_gen(year):

  base = parser.parse("jan 01 "+ str(year))
  date_list = [base + relativedelta(months=x) for x in range(0, 12)]
  return date_list
def get_random_cat(list):
  return list[np.random.randint(0,len(list))]

ACCOUNT_NO_range = range(1, max_num_range)
REVENUE_MONTH_DT_range = range(2016, 2019)
ACCOUNT_TYP_CD_range = ['Residential']
ACCOUNT_STATUS_CD_range = ['Active', 'Final', 'Billed', 'Pending']
PREMISE_NO_range = range(1,max_num_range)
CUSTOMER_NO_range= range(1,max_num_range)
ACCOUNT_CREDIT_GRP_CD_range = ['Exempt', 'Arrears', 'Balance']
ACCOUNT_EPP_IN_range = ['Y','N']
LOCAL_OFFICE_NO_range = ['LO001', 'L0002', 'L0003', 'LO0004']
UTILITY_GROUP_CD_range = ['elec', 'gas']
# ZIP_CD = fake.zi['elec', 'gas']
ACCOUNT_NO_list = []
REVENUE_MONTH_DT_list = []
ACCOUNT_TYP_CD_list = []
ACCOUNT_STATUS_CD_list = []
PREMISE_NO_list = []
CUSTOMER_NO_list= []
ACCOUNT_CREDIT_GRP_CD_list = []
ACCOUNT_EPP_IN_list = []
LOCAL_OFFICE_NO_list = []
UTILITY_GROUP_CD_list = []

ZIP_CD_list=[]
ELECTRIC_BILLED_KWH_QT_list=[]
GAS_BILLED_THERMS_QT_list =[]
GAS_BILLED_AM_list =[]
PREMISE_NO_list=[]
ELECTRIC_BILLED_AM_list = []
UNMETERED_BILLED_AM_list = []
TOTAL_BILLED_AM_list = []

for acc_no in ACCOUNT_NO_range:
  zipcode=fake.zipcode()
  ACCOUNT_TYP_CD_val = get_random_cat(ACCOUNT_TYP_CD_range)
  ACCOUNT_STATUS_CD_val = get_random_cat(ACCOUNT_STATUS_CD_range)
  ACCOUNT_CREDIT_GRP_CD_val=get_random_cat(ACCOUNT_CREDIT_GRP_CD_range)
  ACCOUNT_EPP_IN_val = get_random_cat(ACCOUNT_EPP_IN_range)
  LOCAL_OFFICE_NO_val = get_random_cat(LOCAL_OFFICE_NO_range)
  UTILITY_GROUP_CD_val = get_random_cat(UTILITY_GROUP_CD_range)
  for year in REVENUE_MONTH_DT_range:
    REVENUE_MONTH_DT_list=REVENUE_MONTH_DT_list+ date_gen(year)
    ELECTRIC_BILLED_KWH_QT_list =ELECTRIC_BILLED_KWH_QT_list+ bill_gen(5000,ele_pattern)
    ELECTRIC_BILLED_AM_list = ELECTRIC_BILLED_AM_list+bill_gen(600,ele_pattern)
    GAS_BILLED_AM_list =GAS_BILLED_AM_list+bill_gen(500,gas_pattern)

    GAS_BILLED_THERMS_QT_list =GAS_BILLED_THERMS_QT_list+bill_gen(100,gas_pattern)
    UNMETERED_BILLED_AM_list =UNMETERED_BILLED_AM_list+bill_gen(59,gas_pattern)
    TOTAL_BILLED_AM_list= TOTAL_BILLED_AM_list+bill_gen(1200,ele_pattern)
    
    for i in range(0,12):
      ACCOUNT_NO_list.append('a'+format(acc_no,"06"))
      CUSTOMER_NO_list.append('c'+format(acc_no,"06"))
      PREMISE_NO_list.append('pr'+format(acc_no,"06"))
      ZIP_CD_list.append(zipcode)
      ACCOUNT_TYP_CD_list.append(ACCOUNT_TYP_CD_val)
      ACCOUNT_STATUS_CD_list.append(ACCOUNT_STATUS_CD_val)
      ACCOUNT_CREDIT_GRP_CD_list.append(ACCOUNT_CREDIT_GRP_CD_val)
      ACCOUNT_EPP_IN_list.append(ACCOUNT_EPP_IN_val)
      LOCAL_OFFICE_NO_list.append(LOCAL_OFFICE_NO_val)
      UTILITY_GROUP_CD_list.append(UTILITY_GROUP_CD_val)
      
data_case3 = {"ACCOUNT_NO": ACCOUNT_NO_list,"CUSTOMER_NO":CUSTOMER_NO_list, "PREMISE_NO":PREMISE_NO_list, "ZIP_CD":ZIP_CD_list, "ACCOUNT_TYP_CD":ACCOUNT_TYP_CD_list, "ACCOUNT_STATUS_CD":ACCOUNT_STATUS_CD_list,"PREMISE_NO":PREMISE_NO_list, "ACCOUNT_CREDIT_GRP_CD":ACCOUNT_CREDIT_GRP_CD_list, "ACCOUNT_EPP_IN":ACCOUNT_EPP_IN_list, "LOCAL_OFFICE_NO":LOCAL_OFFICE_NO_list,"UTILITY_GROUP_CD":UTILITY_GROUP_CD_list, "REVENUE_MONTH_DT":REVENUE_MONTH_DT_list,"ELECTRIC_BILLED_KWH_QT":ELECTRIC_BILLED_KWH_QT_list, "ELECTRIC_BILLED_AM":ELECTRIC_BILLED_AM_list, "GAS_BILLED_AM":GAS_BILLED_AM_list, "GAS_BILLED_THERMS_QT":GAS_BILLED_THERMS_QT_list, "UNMETERED_BILLED_AM":UNMETERED_BILLED_AM_list,"TOTAL_BILLED_AM":TOTAL_BILLED_AM_list }
df_case3 = pd.DataFrame.from_dict(data_case3)

df_case3=spark.createDataFrame(df_case3)
df_case3.write.format("delta").mode("overwrite").saveAsTable("data_case3")
display(spark.sql("select * from data_case3 "))

# COMMAND ----------

#Test data generation for use case 1
from faker import Faker
fake = Faker()
fake.name()
fake.address()
fake.text()
fake.state()

#List of fields for use case 1 from customer
"""
ACCOUNT_NO 	- Customer Account #
CIS_SVC_ORDER_NO - Service Order #
ORDER_DISTRICT_NM - Service Order District 
ENGINEER_DISTRICT_NM 	- Technician  District
LATITUDE_NO 	- Latitude  location for Order #
LOCAL_OFFICE_NO	 - nearest account local office
LOCAL_OFFICE_TX 	- nearest account local office
LONGITUDE_NO 	 - Longitude  location for Order #
ORDER_APPT_FINISH_TS 	- Appointment finishtime 
ORDER_APPT_START_TS 	- Appointment start time 
ORDER_COMPLETED_ON_TIME_FL	- Order Completed on time flag
ORDER_COMPLIANCE_TS 	- order compliance timestamp
ORDER_CREATED_DURATION_NO 	- Order created duration
ORDER_CREATED_HR	- order created hrs
ORDER_CREATED_TS 	- order created timestamp
ORDER_PENDING_DURATION_NO 	- How Long order pending 
ORDER_PENDING_HR 	- Hrs value - How Long order pending 
ORDER_PENDING_TS	- How Long order pending - timestamp
ORDER_SCHED_TS 	- order scheduled timestamp
ORDER_DISPATCH_DURATION_NO 	- order dispatch duration
ORDER_DISPATCH_HR	- order dispatch duration -hr. value
ORDER_DISPATCH_TS 	- order dispatch duration -timestamp
ORDER_TRAVEL_DURATION_NO 	- How long it takes to travel to Order location
ORDER_TRAVEL_HR 	- hr. value -How long it takes to travel to Order location
ORDER_TRAVEL_TS	 - How long it takes to travel to Order location -timestamp
ORDER_ONSITE_DURATION_NO 	
ORDER_ONSITE_HR 	
ORDER_ONSITE_TS	
ORDER_FINAL_STATE_TS 	
ORDER_FINAL_STATE_DT	
ORDER_FINAL_STATE_HR 	
ORDER_DUE_TS 	
ORDER_PRIORITY_NO 	 - priority
ORDER_STATUS_NM	 - Status
ORDER_TYP_CD	- order type code
ORDER_TYP_NM 	 - Order type ID
PREMISE_NO 	- Premise no 
READ_ROUTE_NO 	
REGION_NM	- Region Number
UTIL_TYP_NM 	- GAS or Electric
WORK_TYPE_CD	- Worktype code
WORK_TYP_NM 	- Worktype ID
TASK_STD_DURATION_ID	
ASSIGNED_TO 	- technician name
APPOINTMENT_IN	
"""

import pandas as pd
import random
from dateutil import parser
import datetime
from dateutil.relativedelta import relativedelta
import datetime

max_num_range = 100

#Numerical
def bill_gen(total_mean, pattern):
  month_mean = total_mean/sum(pattern)
  return [month_mean*(1+np.random.randint(-1,1)/2)*month for month in pattern]

#Date
def date_gen(year):
  base = parser.parse("jan 01 "+ str(year))
  date_list = [base + relativedelta(months=x) for x in range(0, 12)]
  return date_list

#Categorical 
def get_random_cat(list):
  return list[np.random.randint(0,len(list))]


ACCOUNT_NO_range = range(1, max_num_range)
CIS_SVC_ORDER_NO_range = range(1, max_num_range)
ORDER_DISTRICT_NM_range = range(1, max_num_range)
ENGINEER_DISTRICT_NM_range = range(1, max_num_range)
LATITUDE_NO_range = range(1, max_num_range)
LOCAL_OFFICE_NO_range = ['LO001', 'L0002', 'L0003', 'LO0004']
LOCAL_OFFICE_TX_range = ['TXO001', 'TX0002', 'L0003', 'TXO0004']
LONGITUDE_NO_range = range(1, max_num_range)

ORDER_APPT_FINISH_TS_range = range(1, max_num_range)
ORDER_APPT_START_TS_range = range(1, max_num_range)
ORDER_COMPLETED_ON_TIME_FL_range = range(1, max_num_range)
ORDER_COMPLIANCE_TS_range = range(1, max_num_range)
ORDER_CREATED_DURATION_NO_range = range(1, max_num_range)
ORDER_CREATED_HR_range = range(1, max_num_range)
ORDER_CREATED_TS_range = range(1, max_num_range)
ORDER_PENDING_DURATION_NO_range = range(1, max_num_range)
ORDER_PENDING_HR_range = range(1, max_num_range)
ORDER_PENDING_TS_range = range(1, max_num_range)
ORDER_SCHED_TS_range = range(1, max_num_range)

ORDER_DISPATCH_DURATION_NO_range = [0,1,2,3,4,5]
ORDER_DISPATCH_HR_range = [0,1,2,3,4,5]
ORDER_DISPATCH_TS_range = [0,1,2,3,4,5]

ORDER_TRAVEL_DURATION_NO_range = [5,6,7,8,9,10]
ORDER_TRAVEL_HR_range = [5,6,7,8,9,10]
ORDER_TRAVEL_TS_range = [5,6,7,8,9,10]

ORDER_ONSITE_DURATION_NO_range = [20,30,40]
ORDER_ONSITE_HR_range = [20,30,40]
ORDER_ONSITE_TS_range = [20,30,40]

ORDER_FINAL_STATE_TS_range = ['Ready','Delivered','Dispatched']
ORDER_FINAL_STATE_DT_range = range(2016, 2019)
ORDER_FINAL_STATE_HR_range = ['Ready','Delivered','Dispatched']
ORDER_DUE_TS_range = range(1, max_num_range)

ORDER_PRIORITY_NO_range = ['Low','Medium','High']
ORDER_STATUS_NM_range = ['Open','In-progress','Closed']
ORDER_TYP_CD_range = ['Open','In-progress','Closed']
ORDER_TYP_NM_range = ['Open','In-progress','Closed']

PREMISE_NO_range = range(1, max_num_range)
READ_ROUTE_NO_range = range(1, max_num_range)
REGION_NM_range = range(1, max_num_range)
UTIL_TYP_NM_range = range(1, max_num_range)
WORK_TYPE_CD_range = range(1, max_num_range)
WORK_TYP_NM_range = range(1, max_num_range)
TASK_STD_DURATION_ID_range = range(1, max_num_range)
ASSIGNED_TO_range = range(1, max_num_range)
APPOINTMENT_IN_range = range(1, max_num_range)


ACCOUNT_NO_list = []
CIS_SVC_ORDER_NO_list = []
ORDER_DISTRICT_NM_list = []
ENGINEER_DISTRICT_NM_list = []
LATITUDE_NO_list = []
LOCAL_OFFICE_NO_list = [] 
LOCAL_OFFICE_TX_list = [] 
LONGITUDE_NO_list = []

ORDER_APPT_FINISH_TS_list = []
ORDER_APPT_START_TS_list = []
ORDER_COMPLETED_ON_TIME_FL_list = []
ORDER_COMPLIANCE_TS_list = []
ORDER_CREATED_DURATION_NO_list = []
ORDER_CREATED_HR_list = []
ORDER_CREATED_TS_list = []
ORDER_PENDING_DURATION_NO_list = []
ORDER_PENDING_HR_list = []
ORDER_PENDING_TS_list = []
ORDER_SCHED_TS_list = []

ORDER_DISPATCH_DURATION_NO_list = [] 
ORDER_DISPATCH_HR_list = [] 
ORDER_DISPATCH_TS_list = [] 

ORDER_TRAVEL_DURATION_NO_list = [] 
ORDER_TRAVEL_HR_list = [] 
ORDER_TRAVEL_TS_list = [] 

ORDER_ONSITE_DURATION_NO_list = [] 
ORDER_ONSITE_HR_list = [] 
ORDER_ONSITE_TS_list = [] 

ORDER_FINAL_STATE_TS_list = [] 
ORDER_FINAL_STATE_DT_list = []
ORDER_FINAL_STATE_HR_list = [] 
ORDER_DUE_TS_list = []

ORDER_PRIORITY_NO_list = [] 
ORDER_STATUS_NM_list = [] 
ORDER_TYP_CD_list = [] 
ORDER_TYP_NM_list = [] 

PREMISE_NO_list = []
READ_ROUTE_NO_list = []
REGION_NM_list = []
UTIL_TYP_NM_list = []
WORK_TYPE_CD_list = []
WORK_TYP_NM_list = []
TASK_STD_DURATION_ID_list = []
ASSIGNED_TO_list = []
APPOINTMENT_IN_list = []


# for loop 

for acc_no in ACCOUNT_NO_range:
   ACCOUNT_NO_list.append(1)
  

data_case1 = {
              "ACCOUNT_NO": ACCOUNT_NO_list,
              "CIS_SVC_ORDER_NO ":CIS_SVC_ORDER_NO_list, 
              "ORDER_DISTRICT_NM":ORDER_DISTRICT_NM_list, 
              "ENGINEER_DISTRICT_NM ":ENGINEER_DISTRICT_NM_list, 
              "LATITUDE_NO ":LATITUDE_NO_list, 
              "LOCAL_OFFICE_NO":LOCAL_OFFICE_NO_list,
              "LOCAL_OFFICE_TX":LOCAL_OFFICE_TX_list, 
              "LONGITUDE_NO":LONGITUDE_NO_list, 
              
              "ORDER_APPT_FINISH_TS":ORDER_APPT_FINISH_TS_list, 
              "ORDER_APPT_START_TS":ORDER_APPT_START_TS_list,
              
              "ORDER_COMPLETED_ON_TIME_FL":ORDER_COMPLETED_ON_TIME_FL_list, 
              "ORDER_COMPLIANCE_TS":ORDER_COMPLIANCE_TS_list,
              
              "ORDER_CREATED_DURATION_NO":ORDER_CREATED_DURATION_NO_list, 
              "ORDER_CREATED_HR":ORDER_CREATED_HR_list, 
              "ORDER_CREATED_TS":ORDER_CREATED_TS_list, 
              
              "ORDER_PENDING_DURATION_NO":ORDER_PENDING_DURATION_NO_list, 
              "ORDER_PENDING_HR":ORDER_PENDING_HR_list,
              "ORDER_PENDING_TS":ORDER_PENDING_TS_list,
              
               "ORDER_SCHED_TS": ORDER_SCHED_TS_list ,
               "ORDER_DISPATCH_DURATION_NO ": ORDER_DISPATCH_DURATION_NO_list,
               "ORDER_DISPATCH_HR": ORDER_DISPATCH_HR_list,
               "ORDER_DISPATCH_TS": ORDER_DISPATCH_TS_list,
               "ORDER_TRAVEL_DURATION_NO": ORDER_TRAVEL_DURATION_NO_list,
               "ORDER_TRAVEL_HR": ORDER_TRAVEL_HR_list,
               "ORDER_TRAVEL_TS": ORDER_TRAVEL_TS_list,
               "ORDER_ONSITE_DURATION_NO": ORDER_ONSITE_DURATION_NO_list,
               "ORDER_ONSITE_HR": ORDER_ONSITE_HR_list,
               "ORDER_ONSITE_TS": ORDER_ONSITE_TS_list,
               "ORDER_FINAL_STATE_TS": ORDER_FINAL_STATE_TS_list,
                 
               "ORDER_FINAL_STATE_DT": ORDER_FINAL_STATE_DT_list,
               "ORDER_FINAL_STATE_HR": ORDER_FINAL_STATE_HR_list,
               "ORDER_DUE_TS": ORDER_DUE_TS_list,
               "ORDER_PRIORITY_NO": ORDER_PRIORITY_NO_list,
               "ORDER_STATUS_NM": ORDER_STATUS_NM_list,
               "ORDER_TYP_CD": ORDER_TYP_CD_list,
               "ORDER_TYP_NM": ORDER_TYP_NM_list,
  
               "PREMISE_NO": PREMISE_NO_list,
               "READ_ROUTE_NO": READ_ROUTE_NO_list,
               "REGION_NM": REGION_NM_list,
  
               "UTIL_TYP_NM ": UTIL_TYP_NM_list,
               "WORK_TYPE_CD": WORK_TYPE_CD_list,
               "WORK_TYP_NM": WORK_TYP_NM_list,
               "TASK_STD_DURATION_ID": TASK_STD_DURATION_ID_list,
               "ASSIGNED_TO": ASSIGNED_TO_list,
               "APPOINTMENT_IN": APPOINTMENT_IN_list
             }

df_case1 = pd.DataFrame.from_dict(data_case1)
df_case1=spark.createDataFrame(df_case1)
display(df_case1)



# COMMAND ----------

ele_pattern = [10,10,8,7,6,5,5,6,6,7,8,9]

def bill_gen(total_mean, pattern):
  month_mean = total_mean/sum(pattern)
  return [month_mean*(1+np.random.randint(-1,1)/2)*month for month in pattern]
bill_gen(500, ele_pattern)
from dateutil import parser
import datetime
from dateutil.relativedelta import relativedelta
import datetime


def date_gen(year):

  base = parser.parse("jan 01 "+ str(year))
  date_list = [base + relativedelta(months=x) for x in range(0, 12)]
  return date_list
date_gen(2014)
np.random.randint

# COMMAND ----------

# MAGIC %md ## Setup data access and loading data

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #### Mounting Blob to Azure Databricks 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #### Loading Data from SQL DW

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #### Loading Data from Blob 

# COMMAND ----------

taxes2013 = spark.read.format("csv").option("header", "true").load("dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv")


# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Exploratory Data Analysis

# COMMAND ----------

display(taxes2013.summary())


# COMMAND ----------

# MAGIC %md ## Data Cleaning and enrichment

# COMMAND ----------

#example of enriching the dataset with temperature forecast for every zip code. 

taxes2013_l = taxes2013.limit(20)
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

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegressionSummary

from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator

from pyspark.sql.functions import when
amazon_req_ds =spark.sql("select (case when (bytes_sent <= 1000000) and (request_time-_tcwait/1000)<4 then 1 else (case when bytes_sent > 1000000 and ((8*bytes_sent)/(1000000*(request_time-_tcwait/1000)))>2 then 1 else 0 end) end) good_count, coserver_id, cast (raw_asn as string) as asn, mbps, counter, player_speed, cast(status_code as string) as status_code, request_time, msec, distr_id,_tcpinfo_rtt, cache_status, cc, state, distr_id_geo,upstream_addr, cast (substr(upstream_response_time, 4) as float) as upstream_response_time, upstream_status, cast (substr(upstream_header_time, 4) as float) as upstream_header_time   from amazon_http_req").na.drop()

amazon_req_ds.registerTempTable("amazon_req_ds")
amazon_req_added =spark.sql("select (case when good_count =1 then 0.004030776357765261 else 1- 0.004030776357765261 end) as class_weight, * from amazon_req_ds")

trainingData, testData = amazon_req_added.randomSplit([0.8,0.2])

# non_na_dataset = trainingData.filter(trainingData.fail_qty !=0)
# for i in range(1,2):
#   non_na_dataset_samples = non_na_dataset.sample(True, 0.9)
#   trainingData = trainingData.union(non_na_dataset_samples)

  # labelIndexer = StringIndexer(inputCol="REPLACE_PN", outputCol="part").fit(dataset) 
ccIndexer = StringIndexer(inputCol="cc", outputCol="cc_n").fit(amazon_req_ds)
stateIndexer = StringIndexer(inputCol="state", outputCol="state_n").fit(amazon_req_ds)
asnIndexer = StringIndexer(inputCol="asn", outputCol="asn_n").fit(amazon_req_ds)
status_codeIndexer = StringIndexer(inputCol="status_code", outputCol="status_code_n").fit(amazon_req_ds)
cache_statusIndexer = StringIndexer(inputCol="cache_status", outputCol="cache_status_n").fit(amazon_req_ds)
distr_id_geoIndexer = StringIndexer(inputCol="distr_id_geo", outputCol="distr_id_geo_n").fit(amazon_req_ds)
distr_id_Indexer = StringIndexer(inputCol="distr_id", outputCol="distr_id_n").fit(amazon_req_ds)
# upstream_statusIndexer = StringIndexer(inputCol="upstream_status", outputCol="upstream_status_n").fit(amazon_req_ds)


# encoder = OneHotEncoderEstimator(inputCols=["cc_n", "state_n","asn_n", "status_code_n", "cache_status_n","distr_id_geo_n","distr_id_n", "upstream_status_n"],
#                                  outputCols=["cc_vec", "state_vec","asn_vec", "status_code_vec", "cache_status_vec","distr_id_geo_vec","distr_id_vec", "upstream_status_vec" ])
encoder = OneHotEncoderEstimator(inputCols=["cc_n", "state_n","asn_n", "status_code_n", "cache_status_n","distr_id_geo_n","distr_id_n"],
                                 outputCols=["cc_vec", "state_vec","asn_vec", "status_code_vec", "cache_status_vec","distr_id_geo_vec","distr_id_vec"])


  
assembler = VectorAssembler(
  inputCols=["cc_vec", "state_vec","asn_vec","status_code_vec", "cache_status_vec","distr_id_geo_vec","distr_id_vec",  "mbps", "counter", "player_speed", "request_time", "msec", "_tcpinfo_rtt"],
  outputCol="features")


# COMMAND ----------

# MAGIC %md ## ML for Classification

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor

from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator

from pyspark.sql.functions import when
mldataset = spark.sql("select COUNTRY, PartNumber, bround(Age) as Age_round, sum(ShipQuantity) as Ship_Qty, sum(cast(fail_qty as float)) Fail_Qty from one_part_dataset_1 where COUNTRY <>'' and PartNumber <>'' group by COUNTRY, PartNumber, Age_round ")

trainingData, testData = mldataset.randomSplit([0.8,0.2])

# non_na_dataset = trainingData.filter(trainingData.fail_qty !=0)
# for i in range(1,2):
#   non_na_dataset_samples = non_na_dataset.sample(True, 0.9)
#   trainingData = trainingData.union(non_na_dataset_samples)

  # labelIndexer = StringIndexer(inputCol="REPLACE_PN", outputCol="part").fit(dataset) 
countryIndexer = StringIndexer(inputCol="COUNTRY", outputCol="cn").fit(mldataset)
productIndexer = StringIndexer(inputCol="PartNumber", outputCol="product").fit(mldataset)

encoder = OneHotEncoderEstimator(inputCols=["cn", "product","Age_round"],
                                 outputCols=["countryVec", "productVec","ageVec"])

assembler = VectorAssembler(
  inputCols=["ageVec","Ship_Qty","countryVec","productVec"],
  outputCol="features")






  # featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dataset4)
  # Split the data into training and test sets (30% held out for testing)
  # Train a RandomForest model.
rf =RandomForestRegressor(labelCol="Fail_Qty", featuresCol="features")

# Convert indexed labels back to original labels.
# labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
#                                labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
rf_pipeline = Pipeline(stages=[countryIndexer,productIndexer,encoder, assembler, rf])

# Train model.  This also runs the indexers.
rf_model = rf_pipeline.fit(trainingData)

# Make predictions.
rf_predictions = rf_model.transform(testData)

rf_predictions.select("prediction", "Fail_Qty", "features").show(50)

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="Fail_Qty", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(rf_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rfModel = rf_model.stages[1]
print(rfModel)  # summary only



# # Select (prediction, true label) and compute test error
# evaluator = MulticlassClassificationEvaluator(
#     labelCol="fail_qty", predictionCol="prediction", metricName="weightedRecall")
# rf_accuracy = evaluator.evaluate(rf_predictions)
# print("Test Error = %g" % (1.0 - rf_accuracy))

# rfModel = rf_model.stages[2]
# print(rfModel)  # summary only

# COMMAND ----------

# MAGIC %md ## ML for Regression

# COMMAND ----------







  # featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(dataset4)
  # Split the data into training and test sets (30% held out for testing)
  # Train a RandomForest model
lr = LogisticRegression(maxIter=300, regParam=0.3, elasticNetParam=0.8,weightCol="class_weight",labelCol="good_count",)
  
  
# rf =RandomForestClassifier(labelCol="good_count", featuresCol="features", numTrees= 30)

# Convert indexed labels back to original labels.
# labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
#                                labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
lr_pipeline = Pipeline(stages=[ccIndexer,stateIndexer,asnIndexer,status_codeIndexer,cache_statusIndexer,distr_id_geoIndexer,distr_id_Indexer,encoder, assembler, lr])

# Train model.  This also runs the indexers.
lr_model = lr_pipeline.fit(trainingData)

# Make predictions.
lr_predictions = lr_pipeline.transform(testData)

# rf_predictions.select("prediction", "Fail_Qty", "features").show(50)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="good_count", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(lr_predictions)
print("Test Error = %g" % (1.0 - accuracy))

lrModel = lr_model.stages[-1]
print(lrModel)  # summary only




# # Select (prediction, true label) and compute test error
# evaluator = MulticlassClassificationEvaluator(
#     labelCol="fail_qty", predictionCol="prediction", metricName="weightedRecall")
# rf_accuracy = evaluator.evaluate(rf_predictions)
# print("Test Error = %g" % (1.0 - rf_accuracy))

# rfModel = rf_model.stages[2]
# print(rfModel)  # summary only

# COMMAND ----------

# MAGIC %md ## Timeseries forecast

# COMMAND ----------

# MAGIC %md ### Data prep specific to timeseries

# COMMAND ----------

import pandas as pd
import numpy as np
from random import shuffle
output_seq_len =12
input_seq_len =12

class TS_Dataset:
  columns=None
  def __init__(self,spark_df, batch_size, entity_field,cat_fields, input_fields, output_fields, input_seq_len,output_seq_len, standardized):
    self.input_seq =[] 
    self.output_seq =[]
    self.entity_counter=0
    self.spark_df = spark_df
    self.batch_size = batch_size
    self.entity_field = entity_field
    self.cat_fields=cat_fields
    self.input_fields= input_fields
    self.output_fields=output_fields
    self.input_seq_len = input_seq_len
    self.output_seq_len=output_seq_len
    self.standardized =standardized
    self.pd_df =  spark_df.toPandas()
    self.pd_df_inputs, self.pd_df_outputs,self.std, self.mean, columns = self.encode_ts_data(self.pd_df,self.columns, cat_fields, input_fields, output_fields, standardized)
    if TS_Dataset.columns is None:
        TS_Dataset.columns = columns
    self.batching_data()
  def add_missing_dummy_columns( self,d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0
  def fix_columns( self, d, columns ):  

      self.add_missing_dummy_columns( d, columns )

      # make sure we have all the columns we need
      assert( set( columns ) - set( d.columns ) == set())

      extra_cols = set( d.columns ) - set( columns )
      if extra_cols:
          print("extra columns:", extra_cols)

      d = d[ columns ]
      return d
  def encode_ts_data(self, df,columns, cat_fields, input_fields, output_fields, standardized=True):

    for cat_field in cat_fields:
      temp = pd.get_dummies(self.pd_df[cat_field], prefix=cat_field)
      self.pd_df = pd.concat([self.pd_df, temp], axis = 1)
      del  temp
    if columns is not None:
      self.fix_columns(self.pd_df,columns)
    pd_df_inputs = self.pd_df[input_fields].values.copy()
    pd_df_outputs = self.pd_df[output_fields].values.copy()
    std=[]
    mean=[]
    encoded_input_fields = ["encoded_"+field for field in input_fields]
    encoded_output_fields = ["encoded_"+field for field in output_fields]
    if standardized:
      for i,label in zip(range(pd_df_inputs.shape[1]),encoded_input_fields) :
        temp_mean = pd_df_inputs[:, i].mean()
        temp_std = pd_df_inputs[:, i].std()
        pd_df_inputs[:, i] = (pd_df_inputs[:, i] - temp_mean) / temp_std
        self.pd_df[label]= pd_df_inputs[:, i]
      for i,label in zip(range(pd_df_outputs.shape[1]),encoded_output_fields) :
        temp_mean = pd_df_outputs[:, i].mean()
        temp_std = pd_df_outputs[:, i].std()
        std.append(temp_std)
        mean.append(temp_mean)
        pd_df_outputs[:, i] = (pd_df_outputs[:, i] - temp_mean) / temp_std
        self.pd_df[label]= pd_df_outputs[:, i]


    return pd_df_inputs, pd_df_outputs,std, mean, self.pd_df.columns
  
  def batching_data(self):
    
    entity_list = np.unique(self.pd_df[self.entity_field])
    encoded_input_fields = ["encoded_"+field for field in self.input_fields]
    encoded_output_fields = ["encoded_"+field for field in self.output_fields]
    for entity in entity_list:
      entity_df = self.pd_df[self.pd_df[self.entity_field] == entity]
      x =entity_df[encoded_input_fields].values
      y =entity_df[encoded_output_fields].values
      start_point = len(entity_df) - self.input_seq_len - self.output_seq_len

      start_x_idx = range(start_point)
      input_batch_idxs = [list(range(i, i+self.input_seq_len)) for i in start_x_idx]
      output_batch_idxs = [list(range(i+self.input_seq_len, i+self.input_seq_len+self.output_seq_len)) for i in start_x_idx]
      for input_batch_id in input_batch_idxs:
        self.input_seq.append(np.take(x,input_batch_id, axis = 0))
      for output_batch_id in output_batch_idxs:
        self.output_seq.append(np.take(y,output_batch_id, axis = 0))
    zip_list = list(zip(self.output_seq, self.input_seq))
    shuffle(zip_list)
    self.output_seq, self.input_seq = zip(*zip_list)
    self.output_seq = np.array(self.output_seq)
    self.input_seq = np.array(self.input_seq)
  
  def next_batch(self):
    
    if self.entity_counter== len(self.input_seq):
      #last entity reach, continue checking for entity counter
      
      raise ValueError("End of data reached at:",len(self.input_seq))
    
    if self.entity_counter+self.batch_size <=len(self.input_seq):
      self.entity_counter=self.entity_counter+self.batch_size
      return self.input_seq[self.entity_counter:self.entity_counter+self.batch_size],self.output_seq[self.entity_counter:self.entity_counter+self.batch_size]   
    else:
      self.entity_counter = len(self.input_seq)
      return self.input_seq[self.entity_counter:],self.output_seq[self.entity_counter:]      

    
  def one_shot_prediction(self):
    
    entity_list = np.unique(self.pd_df[self.entity_field])
    encoded_input_fields = ["encoded_"+field for field in self.input_fields]
    encoded_output_fields = ["encoded_"+field for field in self.output_fields]
    input_seq =[]
    output_seq = []
    output_df=[]
    for entity in entity_list:
      entity_df = self.pd_df[self.pd_df[self.entity_field] == entity]
      x =entity_df[encoded_input_fields].values
      y =entity_df[encoded_output_fields].values
      start_point = len(entity_df) - self.input_seq_len - self.output_seq_len
      print(start_point)
      start_x_idx = range(start_point)
      input_batch_idxs = [list(range(i, i+self.input_seq_len)) for i in start_x_idx]
      output_batch_idxs = [list(range(i+self.input_seq_len, i+self.input_seq_len+self.output_seq_len)) for i in start_x_idx]
      for input_batch_id in input_batch_idxs:
        input_seq.append(np.take(x,input_batch_id, axis = 0))
      for output_batch_id in output_batch_idxs:
        output_seq.append(np.take(y,output_batch_id, axis = 0))
      output_df.append(entity_df[start_point:])
    init_df = output_df[0]
    for i in range(1,len(output_df)):
      init_df =init_df.append(output_df[i],ignore_index=True)

    return np.array(input_seq), np.array(output_seq), init_df

 

cat_fields = ['ACCOUNT_STATUS_CD','ACCOUNT_CREDIT_GRP_CD','ACCOUNT_EPP_IN','LOCAL_OFFICE_NO','UTILITY_GROUP_CD','ZIP_CD']
input_fields = ['ELECTRIC_BILLED_AM','ELECTRIC_BILLED_KWH_QT', 'GAS_BILLED_THERMS_QT','GAS_BILLED_AM','UNMETERED_BILLED_AM','TOTAL_BILLED_AM']
output_fields  = ['ELECTRIC_BILLED_AM','ELECTRIC_BILLED_KWH_QT', 'GAS_BILLED_THERMS_QT','GAS_BILLED_AM','UNMETERED_BILLED_AM','TOTAL_BILLED_AM']
df_case3 =spark.sql("select * from data_case3 order by ACCOUNT_NO, REVENUE_MONTH_DT ")
df_case3_train =spark.sql("select * from data_case3 where  ACCOUNT_NO !='a000001' or ACCOUNT_NO !='a000002' and ACCOUNT_NO !='a000003' or ACCOUNT_NO !='a000004' or ACCOUNT_NO !='a000005' or ACCOUNT_NO !='a000006'   order by ACCOUNT_NO, REVENUE_MONTH_DT")
df_case3_val =spark.sql("select * from data_case3 where  ACCOUNT_NO ='a000003' or ACCOUNT_NO ='a000004' or ACCOUNT_NO ='a000005' or ACCOUNT_NO ='a000006'   order by ACCOUNT_NO, REVENUE_MONTH_DT")


df_case3_test = spark.sql("select * from data_case3 where  ACCOUNT_NO ='a000001' or ACCOUNT_NO ='a000002' order by ACCOUNT_NO, REVENUE_MONTH_DT")
ts_dataset=TS_Dataset(df_case3, 30, 'ACCOUNT_NO',cat_fields, input_fields, output_fields, 12,12, True)
ts_dataset_train = TS_Dataset(df_case3_train, 30, 'ACCOUNT_NO',cat_fields, input_fields, output_fields, 12,12, True)
ts_dataset_val = TS_Dataset(df_case3_val, 60, 'ACCOUNT_NO',cat_fields, input_fields, output_fields, 12,12, True)
ts_dataset_test =TS_Dataset(df_case3_test, 60, 'ACCOUNT_NO',cat_fields, input_fields, output_fields, 12,12, True)


# COMMAND ----------

#function to encode timeseries spark dataframe into dataset ready to be consumed by timeseries model. 

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

input_seq_len = 24
output_seq_len = 12
def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0
def fix_columns( d, columns ):  

    add_missing_dummy_columns( d, columns )

    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print("extra columns:", extra_cols)

    d = d[ columns ]
    return d
def encode_ts_data(sparkdf,columns, cat_fields, input_fields, output_fields, standardized=True):
  pd_df =  sparkdf.toPandas()

  for cat_field in cat_fields:
    temp = pd.get_dummies(pd_df[cat_field], prefix=cat_field)
    pd_df = pd.concat([pd_df, temp], axis = 1)
    del pd_df[cat_field], temp
  if columns is not None:
    fix_columns(pd_df,columns)
  pd_df_inputs = pd_df[input_fields].values.copy()
  pd_df_outputs = pd_df[output_fields].values.copy()
  std=[]
  mean=[]
  if standardized:
    for i in range(pd_df_inputs.shape[1]):
      temp_mean = pd_df_inputs[:, i].mean()
      temp_std = pd_df_inputs[:, i].std()
      pd_df_inputs[:, i] = (pd_df_inputs[:, i] - temp_mean) / temp_std
    for i in range(pd_df_outputs.shape[1]):
      temp_mean = pd_df_outputs[:, i].mean()
      temp_std = pd_df_outputs[:, i].std()
      std.append(temp_std)
      mean.append(temp_mean)
      pd_df_outputs[:, i] = (pd_df_outputs[:, i] - temp_mean) / temp_std
  return pd_df_inputs, pd_df_outputs,std, mean, pd_df.columns


def generate_train_samples(x, y, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
 
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)
 
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
 
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
 
    return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)
 
def generate_test_samples(x, y, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
 
    total_samples = x.shape[0]+1
 
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
 
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
 
    return input_seq, output_seq

#####



    


# COMMAND ----------

# MAGIC %md ### Graph definition

# COMMAND ----------

from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import tensorflow as tf
import copy
import os

## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003  

## Network Parameters
# length of input signals
input_seq_len = input_seq_len
# length of output signals
output_seq_len = output_seq_len
# size of LSTM Cell
hidden_dim = 96 
# num of input signals
input_dim = len(input_fields)
# num of output signals
output_dim = len(output_fields)
# num of stacked lstm layers 
num_stacked_layers = 2
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5 

def build_graph(feed_previous = False):
    
    tf.reset_default_graph()
    
    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    
    weights = {
        'out': tf.get_variable('Weights_out', \
                               shape = [hidden_dim, output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out', \
                               shape = [output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.constant_initializer(0.)),
    }
                                          
    with tf.variable_scope('Seq2seq'):
        # Encoder: inputs
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
               for t in range(input_seq_len)
        ]

        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(output_seq_len)
        ]

        # Give a "GO" token to the decoder. 
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the 
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

        with tf.variable_scope('LSTMCell'): 
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)
         
        def _rnn_decoder(decoder_inputs,
                        initial_state,
                        cell,
                        loop_function=None,
                        scope=None):
          """RNN decoder for the sequence-to-sequence model.
          Args:
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            initial_state: 2D Tensor with shape [batch_size x cell.state_size].
            cell: rnn_cell.RNNCell defining the cell function and size.
            loop_function: If not None, this function will be applied to the i-th output
              in order to generate the i+1-st input, and decoder_inputs will be ignored,
              except for the first element ("GO" symbol). This can be used for decoding,
              but also for training to emulate http://arxiv.org/abs/1506.03099.
              Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
            scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing generated outputs.
              state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                 states can be the same. They are different for LSTM cells though.)
          """
          with variable_scope.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
              if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                  inp = loop_function(prev, i)
              if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
              output, state = cell(inp, state)
              outputs.append(output)
              if loop_function is not None:
                prev = output
          return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              feed_previous,
                              dtype=dtypes.float32,
                              scope=None):
          """Basic RNN sequence-to-sequence model.
          This model first runs an RNN to encode encoder_inputs into a state vector,
          then runs decoder, initialized with the last encoder state, on decoder_inputs.
          Encoder and decoder use the same RNN cell type, but don't share parameters.
          Args:
            encoder_inputs: A list of 2D Tensors [batch_size x input_size].
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            feed_previous: Boolean; if True, only the first of decoder_inputs will be
              used (the "GO" symbol), all other inputs will be generated by the previous 
              decoder output using _loop_function below. If False, decoder_inputs are used 
              as given (the standard decoder case).
            dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
            scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
              state: The state of each decoder cell in the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
          """
          with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
            enc_cell = copy.deepcopy(cell)
            _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
            return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function), _rnn_decoder(decoder_inputs, enc_state, cell)
#             if feed_previous:
#                 return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
#             else:
#                 return _rnn_decoder(decoder_inputs, enc_state, cell)

        def _loop_function(prev, _):
          '''Naive implementation of loop function for _rnn_decoder. Transform prev from 
          dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
          used as decoder input of next time step '''
          return tf.matmul(prev, weights['out']) + biases['out']
        
        a,b = _basic_rnn_seq2seq(
            enc_inp, 
            dec_inp, 
            cell, 
            feed_previous = feed_previous
        )
        
        dec_outputs_pred, _ =a
        dec_outputs, dec_memory =b
#         dec_outputs_pred, _ = _basic_rnn_seq2seq(
#             enc_inp, 
#             dec_inp, 
#             cell, 
#             feed_previous = True
#         )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]
        reshaped_outputs_pred = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs_pred]

        
    # Training loss and optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 regularization for weights and biases
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)
        
    saver = tf.train.Saver
    
    return dict(
        enc_inp = enc_inp, 
        target_seq = target_seq, 
        train_op = optimizer, 
        loss=loss,
        saver = saver, 
        reshaped_outputs = reshaped_outputs,
        reshaped_outputs_pred = reshaped_outputs_pred
        )

# COMMAND ----------

# MAGIC %md ### Run training for timeseries

# COMMAND ----------

total_iteractions = 2000
batch_size = 40
KEEP_RATE = 0.5
train_losses = []
val_losses = []
check_point = 500


rnn_model = build_graph(feed_previous=False)

val_x, val_y,_ = ts_dataset_val.one_shot_prediction()

init = tf.global_variables_initializer()
best_mse=0
with tf.Session() as sess:

    sess.run(init)
    #Restore from previously trained best model
#     saver = rnn_model['saver']().restore(sess,  os.path.join('./', 'total_bytes_forecast'))

    print("Training losses: ")
    for i in range(total_iteractions):
        try:
          batch_input, batch_output = ts_dataset_train.next_batch()
        except:
          ts_dataset_train = TS_Dataset(df_case3_train, 30, 'ACCOUNT_NO',cat_fields, input_fields, output_fields, 12,12, True)
          batch_input, batch_output = ts_dataset_train.next_batch()
          
        
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)

        if i%100==0:
          print("Step {}, loss {}".format(i,loss_t))
        if i%check_point ==0 and i>0:
          
          temp_saver = rnn_model['saver']()
          save_path = temp_saver.save(sess, os.path.join('./', 'case3_forecast'))
          feed_dict = {rnn_model['enc_inp'][t]: val_x[:, t, :] for t in range(input_seq_len)} # batch prediction
          feed_dict.update({rnn_model['target_seq'][t]: np.zeros([val_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
          final_preds = sess.run(rnn_model['reshaped_outputs_pred'], feed_dict)
          final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
          final_preds = np.concatenate(final_preds, axis = 1)
          test_mse = np.mean((final_preds - val_y)**2)
                             
          if (best_mse==0) or (test_mse<best_mse):
             print("Found best model at {} with MSE {}".format(i,test_mse))
             best_mse= test_mse
             temp_saver = rnn_model['saver']()
             save_path = temp_saver.save(sess, os.path.join('./', 'best_model/case3_forecast'))
             print("Best Model saved at: ", save_path)
    
           


# COMMAND ----------

# MAGIC %md ### Visualizing result for 12 months prediction

# COMMAND ----------

rnn_model = build_graph()
test_x, test_y, output_df = ts_dataset_test.one_shot_prediction()
y_test_std = ts_dataset_test.std
y_test_mean = ts_dataset_test.mean

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    
    saver = rnn_model['saver']().restore(sess,  os.path.join('./', 'best_model/case3_forecast'))
    
    feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs_pred'], feed_dict)
    final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
    final_preds = np.concatenate(final_preds, axis = 1)
    test_mse = np.mean((final_preds - test_y)**2)
    print("Test mse is: ", test_mse)


for dim, field_name in enumerate(output_fields):
  
  
## Display forecast for dimension 0 (total_count)
#   test_y_expand = np.concatenate([test_y[:,:,dim][i].reshape(-1) for i in range(0, test_y.shape[0], output_seq_len)], axis = 0)
#   test_y_expand = [item*y_test_std[0] + y_test_mean[dim] for item in test_y_expand]
  final_preds_expand = np.concatenate([final_preds[:,:,dim][i].reshape(-1) for i in [0,11,12,23]], axis = 0)
  final_preds_expand = [item*y_test_std[dim] + y_test_mean[dim] for item in final_preds_expand]

  output_df[field_name+"_predicted"] = final_preds_expand
output_df_spark = spark.createDataFrame(output_df)
display(output_df_spark)
# # ## Display forecast for dimension 0 (total_count)
# # test_y_expand = np.concatenate([test_y[:,:,0][i].reshape(-1) for i in range(0, test_y.shape[0], output_seq_len)], axis = 0)
# # test_y_expand = [item*y_test_std[0] + y_test_mean[0] for item in test_y_expand]
# # final_preds_expand = np.concatenate([final_preds[:,:,0][i].reshape(-1) for i in range(0, final_preds.shape[0], output_seq_len)], axis = 0)
# # final_preds_expand = [item*y_test_std[0] + y_test_mean[0] for item in final_preds_expand]
# # import matplotlib.pyplot as plt
# # plt.clf()
# # plt.plot(final_preds_expand, color = 'orange', label = 'predicted')
# # plt.plot(test_y_expand, color = 'blue', label = 'actual')
# # plt.title("test data - ")
# # plt.legend(loc="upper left")

# # plt.show()
# # display()

# COMMAND ----------

# MAGIC %md ## Scoring and Persisting Result to SQL DW

# COMMAND ----------

# MAGIC %md ### Writing result out to SQL DW using high performance ADB-DW driver

# COMMAND ----------


spark.conf.set(
  "fs.azure.account.key.cdnctllog.blob.core.windows.net",
  "FlD2XbLQBL8BwlibeEoaxI+uI1TfUgnGB5dfGsjTZgX03/8aDDsYCU9Ljn+lc8gDGyk7gAF2ohJW38AEjLlAVg==")

_dw_jdbcHostname = 'cdnctllog.database.windows.net'
_dw_jdbcUsername ='cdnctl'
_dw_jdbcPassword = 'Welcome@123'
# Make sure table has appropriate columns
_tx_dw_table = 'txalarm_anomaly_oct04'


_dw_jdbcDatabase = "cdnctllog"
_dw_jdbcPort = 1433

jdbc_url_dw = 'jdbc:sqlserver://' + _dw_jdbcHostname + ':' + str(_dw_jdbcPort) + ';database=' + _dw_jdbcDatabase + ';user=' + _dw_jdbcUsername + ';password=' + _dw_jdbcPassword


txalarm_anomaly.write \
  .format("com.databricks.spark.sqldw") \
  .option("url", jdbc_url_dw) \
  .mode("overwrite") \
  .option("tempDir", "wasbs://cdnctllog@cdnctllog.blob.core.windows.net/v2") \
  .option("forwardSparkAzureStorageCredentials", "true") \
  .option("dbTable", "dbo."+_tx_dw_table) \
  .option("checkpointLocation", "wasbs://cdnctllog@cdnctllog.blob.core.windows.net/checkpoints/v2").save()
