# Databricks notebook source
# Get iterable file list. Flattens hierarchical folder/file structure.
def GetFilesRecursive(rootPath):
  final_list = []

  for directoryItem in dbutils.fs.ls(rootPath):
    directoryItemPathClean = directoryItem.path.replace("%25", "%").replace("%25", "%")
    
    if directoryItem.isDir() == True:
      final_list = final_list + GetFilesRecursive(directoryItemPathClean)
    else:
      final_list.append(directoryItemPathClean)
  
  return final_list;

# COMMAND ----------

# Delete Spark job residual files (_SUCCESS, _start*, _committed*) down the folder/file hierarchy

import os

def CleanupSparkJobFiles(parquetFolderPath):
  file_paths = GetFilesRecursive(parquetFolderPath)
  
  for file_path in file_paths:
    # Get just the file name
    file_name = os.path.basename(file_path)
    # print(file_name)
    
    if file_name.startswith("_"):
      # Temp job file - delete it
      dbutils.fs.rm(file_path)
    # elif file_name.endswith(".parquet"):
      # Data file - no op
    # else:
      # Something else - no op


# COMMAND ----------

