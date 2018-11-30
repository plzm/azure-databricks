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



# COMMAND ----------

# MAGIC %md ## ML for Classification

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## ML for Regression

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Timeseries forecast

# COMMAND ----------

# MAGIC %md ### Data prep specific to timeseries

# COMMAND ----------


import pandas as pd
## One-hot encode 'state'
states =['AK',
       'AL', 'AR', 'AZ', 'CA', 'CO', 'CT',
       'DC', 'DE', 'FL', 'GA', 'HI', 'IA',
       'ID', 'IL', 'IN', 'KS', 'KY', 'LA',
       'MA', 'MD', 'ME', 'MI', 'MN', 'MO',
       'MS', 'MT', 'NC', 'ND', 'NE', 'NH',
       'NJ', 'NM', 'NV', 'NY', 'OH', 'OK',
       'OR', 'PA', 'RI', 'SC', 'SD', 'TN',
       'TX', 'UT', 'VA', 'VT', 'WA', 'WI',
       'WV', 'WY', 'aol', 'unknown']


local_df_train_pv['state']= local_df_train_pv['state'].astype('category',categories=states)
temp = pd.get_dummies(local_df_train_pv['state'], prefix='state')
local_df_train_pv = pd.concat([local_df_train_pv, temp], axis = 1)
del local_df_train_pv['state'], temp

local_df_test_pv['state']= local_df_test_pv['state'].astype('category',categories=states)
temp = pd.get_dummies(local_df_test_pv['state'], prefix='state')
local_df_test_pv = pd.concat([local_df_test_pv, temp], axis = 1)
del local_df_test_pv['state'], temp

hours = list(range(0,24))

local_df_test_pv['hour']= local_df_test_pv['hour'].astype('category',categories=hours)
temp = pd.get_dummies(local_df_test_pv['hour'])
local_df_test_pv = pd.concat([local_df_test_pv, temp], axis = 1)
del local_df_test_pv['hour'], temp


X_train = df_train.loc[:,['aiv_num_rebuffers', 'total_count', 'avg_mbps','hour']].values.copy()
X_test = df_test.loc[:,['aiv_num_rebuffers', 'total_count', 'avg_mbps','hour']].values.copy()
y_train = df_train[['total_count','aiv_num_rebuffers','avg_mbps']].values.copy()
y_test = df_test[['total_count','aiv_num_rebuffers','avg_mbps']].values.copy()
# y_train = df_train[['total_count']].values.copy()
# y_test = df_test[['total_count']].values.copy()
 
## z-score transform x - not including those one-hot columns!
for i in range(X_train.shape[1]):
    temp_mean = X_train[:, i].mean()
    temp_std = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - temp_mean) / temp_std
    X_test[:, i] = (X_test[:, i] - temp_mean) / temp_std
 
## z-score transform y
y_mean = y_train.mean(axis =0)
y_std = y_train.std(axis =0)
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# COMMAND ----------

input_seq_len = 60
output_seq_len = 10
import numpy as np 
def generate_train_samples(x = X_train, y = y_train, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
 
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)
 
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
 
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
 
    return input_seq, output_seq # in shape: (batch_size, time_steps, feature_dim)
 
def generate_test_samples(x = X_test, y = y_test, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
 
    total_samples = x.shape[0]
 
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
 
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
 
    return input_seq, output_seq

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
input_dim = X_train.shape[1]
# num of output signals
output_dim = y_train.shape[1]
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

total_iteractions = 3001
batch_size = 40
KEEP_RATE = 0.5
train_losses = []
val_losses = []
check_point = 500
# x = np.linspace(0, 40, 130)
# train_data_x = x[:110]

rnn_model = build_graph(feed_previous=False)


init = tf.global_variables_initializer()
best_mse=0
with tf.Session() as sess:

    sess.run(init)
    #Restore from previously trained best model
#     saver = rnn_model['saver']().restore(sess,  os.path.join('./', 'total_bytes_forecast'))

    print("Training losses: ")
    for i in range(total_iteractions):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size)
        
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
#         if best_mse==0:
  
#           feed_dict_pred = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
#           feed_dict_pred.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
#           final_preds = sess.run(rnn_model['reshaped_outputs_pred'], feed_dict_pred)
#           final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
#           final_preds = np.concatenate(final_preds, axis = 1)
#           best_mse = np.mean((final_preds - test_y)**2)
#           print("get initial mse: {}".format(best_mse))
        if i%100==0:
          print("Step {}, loss {}".format(i,loss_t))
        if i%check_point ==0:
          
          temp_saver = rnn_model['saver']()
          save_path = temp_saver.save(sess, os.path.join('./', 'total_bytes_forecast'))
          feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
          feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
          final_preds = sess.run(rnn_model['reshaped_outputs_pred'], feed_dict)
          final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
          final_preds = np.concatenate(final_preds, axis = 1)
          test_mse = np.mean((final_preds - test_y)**2)
                             
          if (best_mse==0) or (test_mse<best_mse):
             print("Found best model at {} with MSE {}".format(i,test_mse))
             best_mse= test_mse
             temp_saver = rnn_model['saver']()
             save_path = temp_saver.save(sess, os.path.join('./', 'best_model/total_bytes_forecast'))
             print("Best Model saved at: ", save_path)
    
           


# COMMAND ----------

# MAGIC %md ### Visualizing result

# COMMAND ----------

## Display forecast for dimension 0 (total_count)
test_y_expand = np.concatenate([test_y[:,:,0][i].reshape(-1) for i in range(0, test_y.shape[0], output_seq_len)], axis = 0)
test_y_expand = [item*y_std[0] + y_mean[0] for item in test_y_expand]
final_preds_expand = np.concatenate([final_preds[:,:,0][i].reshape(-1) for i in range(0, final_preds.shape[0], output_seq_len)], axis = 0)
final_preds_expand = [item*y_std[0] + y_mean[0] for item in final_preds_expand]
import matplotlib.pyplot as plt
plt.clf()
plt.plot(final_preds_expand[:200], color = 'orange', label = 'predicted')
plt.plot(test_y_expand[:200], color = 'blue', label = 'actual')
plt.title("test data - ")
plt.legend(loc="upper left")

plt.show()
display()

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
