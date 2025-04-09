# Databricks notebook source
# MAGIC %md #Model Deployment + Inference to Delta Lake

# COMMAND ----------

# MAGIC %md ##Loading Configuration File

# COMMAND ----------

# MAGIC %run "/Users/bobifamily@outlook.com/de-bd-project/includes/Configuration"

# COMMAND ----------

# MAGIC %md ##Loading Dependencies

# COMMAND ----------

from pyspark.sql.functions import *
import pyspark.sql.functions as F
#from pyspark.sql.functions import col, regexp_replace, lower, trim, count, desc
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel, IndexToString
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ##User Defined Functions (UDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### UDF to Preprocess Dataframes for Inference

# COMMAND ----------

def preprocessing_udf(posts, PostTypes):
    # Renaming column to prevent confusion when joining tables
    posts = posts.withColumnRenamed("id", "question_id")
    # Removing Columns with extensive amount of Nulls. 
    # 'ParentID' does not apply to questions
    posts = posts.drop('LastEditorDisplayName', 'ParentId')
    # We will only process questions, not any other post type (ex: answers, wikis, and more)
    joined_df = posts.join(PostTypes, posts.PostTypeId == PostTypes.Id, "left")
    df = joined_df.filter(col("Type") == "Question")\
                    .drop("Id")
    
    # Converting the tags' string column into a list of tags
    df = df.withColumn("Tags", split(trim(translate(col("Tags"), "<>", " ")), " "))
    # Filtering out empty body text
    df = df.filter("Body is not null")

    # Renaming column named "Body" to "Text" for model compatibility
    df = df.withColumnRenamed("Body", "Text")
    # Preprocessing of the feature column
    processed_df = df.withColumn('Text', regexp_replace('Text', r"http\S+", ""))\
        .withColumn('Text', regexp_replace('Text', r'<.*?>', ''))\
        .withColumn('Text', regexp_replace('Text', r"[^a-zA-Z]", " "))\
        .withColumn('Text', regexp_replace('Text', r"\s+", " "))\
        .withColumn('Text', lower('Text')) \
        .withColumn('Text', trim('Text')) 
    
    ml_input_df = processed_df.select("question_id", "Text")
    
    return processed_df, ml_input_df


# COMMAND ----------

# MAGIC %md
# MAGIC ###UDF to Make Inference on Batched Data

# COMMAND ----------

def predictions_udf(ml_input_df, ml_model, stringindexer):    

    # Loading the pipeline model
    model = PipelineModel.load(ml_model)

    # Making Predictions
    predictions = model.transform(ml_input_df)

    #predicted = predictions.select(col('question_id'), col('Text'), col('Tags'), col('prediction'))
    predicted = predictions.select(col('question_id'), col('Text'), col('prediction'))

    # Decoding the indexer 
    # Loading the indexer saved in datalake 
    indexer = StringIndexerModel.load(stringindexer)

    # Initializing the IndexToString Converter
    i2s = IndexToString(inputCol = 'prediction', outputCol = 'decoded', labels = indexer.labels)
    converted = i2s.transform(predicted)

    # Display the important columns
    return converted
    

# COMMAND ----------

# MAGIC %md
# MAGIC ##Loading the ML Model, the StringIndexer, the Posts File, and the postTypes File.

# COMMAND ----------

posts = spark.read.parquet("/mnt/debdingestiondl/de-bd-project/Landing/Posts/*")
ml_model = "/mnt/debdingestiondl/de-bd-project/Model/"
stringindexer = "/mnt/debdingestiondl/de-bd-project/StringIndexer/"

# Creating the  schema for posttypes table
from pyspark.sql.types import *

PT_schema = StructType([
    StructField("Id", IntegerType(), True),
    StructField("Type", StringType(), True)
])

# Creating the posttypes dataframe
file_location = "/mnt/debdingestiondl/de-bd-project/ML-Training/PostTypes.txt"

PostTypes = (spark.read
            .option("header", "true")
            .option("sep", ",")
            .schema(PT_schema)
            .csv(file_location)            
            )

# COMMAND ----------

#display(posts.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Predicting Labels

# COMMAND ----------

# the preprocessing_udf function returns 3 dataframes  
processed_df, ml_input_df = preprocessing_udf(posts,PostTypes)

# COMMAND ----------

# Executing predictions_udf
results = predictions_udf(ml_input_df,ml_model,stringindexer)


# COMMAND ----------

#display(results.limit(20))

# COMMAND ----------

display(results.count())

# COMMAND ----------

unique_qid = results.select(countDistinct("question_id"))
unique_qid.show()

# COMMAND ----------

display(processed_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Merging the results with the processed input dataframe

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime

# Rename the columns in 'results' DataFrame
processed_df = processed_df.withColumnRenamed("question_id", "proc_question_id").withColumnRenamed("Text", "proc_Text")


# Perform the join operation
results_processed_df = results.join(processed_df, results.question_id == processed_df.proc_question_id, "left")
# Removing duplicated column that resulted from the merge of both dataframes
results_processed_df = results_processed_df\
                            .drop("proc_question_id","proc_Text")\
                            .withColumn("prediction", results_processed_df["prediction"].cast(IntegerType()))



# COMMAND ----------


# display(final_results_df.limit(20))

# COMMAND ----------

# MAGIC %md ##Batch Datetime Stamping and Shipment to Delta Tables

# COMMAND ----------

# MAGIC %md ###Adding a datetime column to mark the date the batch was submitted for inference 

# COMMAND ----------

batch_processing_date = datetime.now().strftime("%Y_%m_%d_%H")

# COMMAND ----------

# Add a new column with the current date
#final_results_df = final_results_df.withColumn('batch_date_stamp', F.current_date())
final_results_df = results_processed_df.withColumn('batch_datetime_stamp', to_timestamp(lit(batch_processing_date), 'yyyy_MM_dd_HH'))


# COMMAND ----------

display(final_results_df.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Initializing Database and Delta Tables If Don't Exist

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS stack_overflow_proj;

# COMMAND ----------

# MAGIC %md ###DDL for the labeled_qs_ext_dtbl  Delta Table

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS stack_overflow_proj.labeled_qs_ext_dtbl (
# MAGIC   question_id INT,
# MAGIC   Text STRING,
# MAGIC   prediction INT,
# MAGIC   decoded STRING,
# MAGIC   AcceptedAnswerId INT,
# MAGIC   AnswerCount INT,
# MAGIC   CommentCount INT,
# MAGIC   CreationDate TIMESTAMP,
# MAGIC   FavoriteCount INT,
# MAGIC   LastEditDate TIMESTAMP,
# MAGIC   LastEditorUserId INT,
# MAGIC   OwnerUserId INT,
# MAGIC   PostTypeId INT,
# MAGIC   Score FLOAT,
# MAGIC   Tags ARRAY <string>,
# MAGIC   Title STRING,
# MAGIC   ViewCount INT,
# MAGIC   Type STRING,
# MAGIC   batch_datetime_stamp TIMESTAMP  
# MAGIC )
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/mnt/debdingestiondl/de-bd-project/BI/labeled_qs_ext_dtbl'

# COMMAND ----------

# bi_folder_path

# COMMAND ----------

# MAGIC %md ###Saving to Delta Table

# COMMAND ----------

final_results_df.write.format("delta")\
    .mode("append")\
    .option("path", f"{bi_folder_path}/labeled_qs_ext_dtbl/labeled_qs_batch_{batch_processing_date}")\
    .saveAsTable("stack_overflow_proj.labeled_qs_ext_dtbl")

# COMMAND ----------

# bi_folder_path

# COMMAND ----------

# MAGIC %md
# MAGIC ##Most Popular Labels 

# COMMAND ----------

#topics = results.withColumnRenamed('decoded', 'topic').select('topic')

# Aggregating the topics and calculating the total qty of each topic

#topic_qty = topics.groupBy(col('topic')).agg(count('topic').alias('qty')).orderBy(desc('qty'))

#display(topic_qty.limit(20))

# COMMAND ----------


