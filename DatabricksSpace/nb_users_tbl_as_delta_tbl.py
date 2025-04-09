# Databricks notebook source
# MAGIC %md #Users Data To Delta Lake External Table

# COMMAND ----------

# MAGIC %run "/Users/bobifamily@outlook.com/de-bd-project/includes/Configuration"
# MAGIC

# COMMAND ----------

# MAGIC %md ##Dependencies

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import trim, regexp_replace, col, countDistinct, current_timestamp, to_timestamp,lit, year, month
from datetime import datetime

# COMMAND ----------

# MAGIC %md ##Reading Raw Users CSV File to Dataframe

# COMMAND ----------

# Creating the schema for the users table

users_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("Age", IntegerType(), True),
    StructField("CreationDate", DateType(), True),
    StructField("DisplayName", StringType(), True),
    StructField("DownVotes", IntegerType(), True),
    StructField("EmailHash", StringType(), True),
    StructField("Location", StringType(), True),
    StructField("Reputation", IntegerType(), True),
    StructField("UpVotes", IntegerType(), True),
    StructField("Views", IntegerType(), True),
    StructField("WebsiteUrl", StringType(), True),
    StructField("AccountId", IntegerType(), True)
])


# COMMAND ----------

users = (spark.read               
         .option("header", "true")
         .option("sep", ",")
         .schema(users_schema)
         .csv(f"{landing_folder_path}/Users/")
         )

display(users.limit(10))

# COMMAND ----------

# MAGIC %md ##Processing Users 

# COMMAND ----------

def processing_users (users):
    # Dropping Empty Columns
    columns_to_drop = ['Age', 'EmailHash','WebsiteUrl']
    users = users.drop (*columns_to_drop)  

    processing_date = datetime.now().strftime("%Y_%m_%d_%H")
    users = users.withColumn('processing_datetime_stamp', to_timestamp(lit(processing_date), 'yyyy_MM_dd_HH'))\
                    .withColumn('Year', year('CreationDate'))\
                    .withColumn('Month', month('CreationDate'))

    # Rearranging the order of the columns
    users_final_df = users.select("id", "CreationDate", "Year", "Month", "DisplayName", "DownVotes", "Location", "Reputation", "UpVotes", "Views", "AccountId", "processing_datetime_stamp")
    
    return users_final_df
    

# COMMAND ----------

users_final_df = processing_users(users)

# COMMAND ----------

#display(users_final_df.limit(10))

# COMMAND ----------

# column_types = users_final_df.dtypes

# # Iterate over the list of tuples to print the column names and their data types
# for column_name, data_type in column_types:
#     print(f"Column '{column_name}' has data type: {data_type}")

# COMMAND ----------

# MAGIC %md ###Initializing Database and Delta Tables If Don't Exist

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS stack_overflow_proj;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS stack_overflow_proj.users_ext_dtbl (
# MAGIC   id INT,  
# MAGIC   CreationDate DATE,
# MAGIC   DisplayName STRING,  
# MAGIC   DownVotes INT,  
# MAGIC   Location STRING,
# MAGIC   Reputation INT,  
# MAGIC   UpVotes INT,
# MAGIC   Views INT,  
# MAGIC   AccountId INT,
# MAGIC   processing_datetime_stamp TIMESTAMP,
# MAGIC   year INT,
# MAGIC   month INT    
# MAGIC )
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/mnt/debdingestiondl/de-bd-project/BI/users_ext_dtbl'

# COMMAND ----------

# MAGIC %md ##Write data to Delta Lake

# COMMAND ----------


users_final_df.write.format("delta")\
    .mode("overwrite")\
    .option("path", f"{bi_folder_path}/users_ext_dtbl")\
    .saveAsTable("stack_overflow_proj.users_ext_dtbl")

# COMMAND ----------


