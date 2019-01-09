# Databricks notebook source
# MAGIC %md
# MAGIC References:<br>
# MAGIC https://docs.azuredatabricks.net/user-guide/secrets/secrets.html<br>
# MAGIC https://docs.azuredatabricks.net/spark/latest/data-sources/azure/azure-storage.html<br>
# MAGIC https://docs.azuredatabricks.net/user-guide/dbfs-databricks-file-system.html<br>
# MAGIC 
# MAGIC Notes:
# MAGIC DBFS commands are very specific. Some hard-coding/specific formats -required-.
# MAGIC 
# MAGIC Prefer Azure Key Vault for secret storage.<br>
# MAGIC Create Azure Key Vault, store secret there. Then create ADB secret scope.<br>
# MAGIC Then use secret scope and storage acct info to mount storage acct/container to DBFS mount point.

# COMMAND ----------

storageAcctName = "pxbrixsa"
containerName = "hack"

secretScopeName = "pzbrixscope"
secretName = "pzbrixsakey"

mountPoint = "/mnt/" + containerName

# COMMAND ----------

# Explicit version works but hard-codes container and storage account names - have not been able to get this to work with variables and string concat
dbutils.fs.mount(
  source = "wasbs://hack@pzbrixsa.blob.core.windows.net",
  mount_point = mountPoint,
  extra_configs = {"fs.azure.account.key.pzbrixsa.blob.core.windows.net":dbutils.secrets.get(scope = secretScopeName, key = secretName)}
)

# COMMAND ----------

# This (or a version of the above using variables) does not work. Gets java.lang.IllegalArgumentException due to invalid mount source. ??
dbutils.fs.mount(
        "wasbs://{cn}@{san}.blob.core.windows.net"
        .format(
            cn=containerName,
            san=storageAcctName),
        "/mnt/{mn}".format(mn=mountPoint)
    )

# COMMAND ----------

# ls in the newly mounted mount point / in the mounted container
display(dbutils.fs.ls("/mnt/" + containerName))

# COMMAND ----------

# Unmount the storage mount point
dbutils.fs.unmount("/mnt/" + containerName)

# COMMAND ----------

# Refresh mounts on other clusters than the one that ran DBFS commands
dbutils.fs.refreshMounts()

# COMMAND ----------

display(dbutils.fs.ls("/mnt"))

# COMMAND ----------

