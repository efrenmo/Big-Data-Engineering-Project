# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training

# COMMAND ----------

# MAGIC %md ## Dependencies

# COMMAND ----------

# Dependencies
from pyspark.sql.types import *
from pyspark.sql.functions import split, translate, trim, explode, regexp_replace, col, lower, countDistinct
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# COMMAND ----------

# MAGIC %md
# MAGIC ##1. Loading the data into dataframes

# COMMAND ----------

# Creating a Spark Session
from pyspark.sql import SparkSession

spark = (SparkSession
         .builder
         .appName("Table Loading")
         .getorCreate())

sc = spark.SparkContext

# COMMAND ----------

# MAGIC %md
# MAGIC ###1.1. Creating the 'Posts' dataframe

# COMMAND ----------

display(dbutils.fs.mounts())

# COMMAND ----------

display(dbutils.fs.ls("/mnt/debdingestiondl/de-bd-project"))

# COMMAND ----------

display(dbutils.fs.ls("/mnt/debdingestiondl/de-bd-project/ML-Training/Posts/"))

# COMMAND ----------

# MAGIC %md
# MAGIC Reading the parquet files

# COMMAND ----------

file_location = "/mnt/debdingestiondl/de-bd-project/ML-Training/Posts/*"

posts = spark.read \
            .parquet(file_location)

display(posts.limit(20))    

# COMMAND ----------

# Renaming the column id with question_id to avoid conflict when joining with the postType dataframe
# Which also has a column named 'id'
posts = posts.withColumnRenamed("id", "question_id")

# COMMAND ----------



# COMMAND ----------

display(posts.describe())

# COMMAND ----------

# Count the null values in the "PostTypeId" column
null_count = posts.filter(col("PostTypeId").isNull()).count()

# Print the count of null values
print("Count of null values in the 'PostTypeId' column:", null_count)

# COMMAND ----------

# MAGIC %md
# MAGIC ###1.2. Creating the posttypes dataframe

# COMMAND ----------

# Creating the  schema for posttypes table
from pyspark.sql.types import *

PT_schema = StructType([
    StructField("Id", IntegerType(), True),
    StructField("Type", StringType(), True)
])


# Creating the posttypes dataframe
file_location = "/mnt/debdingestiondl/de-bd-project/ML-Training/PostTypes.txt"

postTypes = (spark.read
            .option("header", "true")
            .option("sep", ",")
            .schema(PT_schema)
            .csv(file_location)            
            )
display(postTypes)

# COMMAND ----------

display(postTypes)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###1.3. Creating the users dataframe

# COMMAND ----------

# Creating the schema for the users table
from pyspark.sql.types import *

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

# Creating the users dataframe

file_location = "/mnt/debdingestiondl/de-bd-project/ML-Training/users.csv"

users = (spark.read               
         .option("header", "true")
         .option("sep", ",")
         .schema(users_schema)
         .csv(file_location)
         )

display(users.limit(10))

# COMMAND ----------

# Creating the schema for the users table
from pyspark.sql.types import *
 
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

# Creating the users dataframe
file_location = "/mnt/debdingestiondl/de-bd-project/ML-Training/users.csv"
 
users = (spark.read
 .option("header", "true")
 .option("sep", ",")
 .schema(users_schema)
 .csv(file_location))
 
display(users.limit(20))



# COMMAND ----------

# Save the dataframes for easy retrieval
folder = "/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"

# Saving the 3 tables to mounted data lake
posts.write.parquet(f"{folder}posts.parquet")
postTypes.write.parquet(f"{folder}postTypes.parquet")
users.write.parquet(f"{folder}users.parquet")

# COMMAND ----------

# review the local file system
#display(dbutils.fs.ls("dbfs:/"))
display(dbutils.fs.ls("/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"))


# COMMAND ----------

display(dbutils.fs.ls("dbfs:/"))

# COMMAND ----------

dbutils.fs.rm("dbfs:/temporal", recurse=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ##2. Join tables and filter data

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1. Prepare the necessary libraries and load data

# COMMAND ----------

#from pyspark.sql import SparkSessions
from pyspark.sql.functions import split, translate, trim, explode, regexp_replace, col, lower

# COMMAND ----------

# Creating Spark Session
spark = (SparkSession
    	 .builder
    	 .appName("ML Model")
    	 .getOrCreate())
 
sc = spark.sparkContext

# COMMAND ----------

# MAGIC %md 
# MAGIC **Reading Saved Tables**

# COMMAND ----------

# Reading the tables
# Save the dataframes for easy retrieval
folder = "/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"

posts.read.parquet(f"{folder}posts.parquet")
postTypes.read.parquet(f"{folder}postTypes.parquet")
users.read.parquet(f"{folder}users.parquet")


# COMMAND ----------



# COMMAND ----------

# Return descriptive statistics for the posts DataFrame
display(posts.describe())

# COMMAND ----------

display(users.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.2. Join the tables post and postTypes by its postType Id

# COMMAND ----------

# We need the post and postTypes to train the model

joined_df = posts.join(postTypes, posts.PostTypeId == postTypes.Id)
display(joined_df.limit(20))

# COMMAND ----------

print(joined_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.3. Filter & Format the data 
# MAGIC <br>
# MAGIC In the posttypes table, there is a column called Type which indicates if the posts is a question or an answer. We only need the 'question' entires.
# MAGIC <br>For these rows, we will run machine learning model on the 'Body' column of the 'Posts' table. To tell what topic this post is about. 

# COMMAND ----------

# Filter the dataframe to only include questions post type
df = joined_df.filter(col("Type") == "Question")
display(df.limit(20))

# COMMAND ----------

display(df.describe())

# COMMAND ----------

# Formatting the 'Body' and `Tag` columns for machine learning training
df = (df.withColumn("Tags", split(trim(translate(col("Tags"), "<>", " ")), " "))) # Making a list of the tags      
display(df.limit(10))

# COMMAND ----------

directory = "/mnt/debdingestiondl/de-bd-project/ML-Training/"

df.repartition(1) \
    .write \
    .format("parquet") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .option("compression", "snappy") \
    .save(f"{directory}900questions.parquet")

# COMMAND ----------

# Verify the column types of the DataFrame
df.printSchema()


# COMMAND ----------

# MAGIC %md
# MAGIC The code below is using Apache Spark DataFrame operations to perform data transformations on a DataFrame. Here's a breakdown of the code:
# MAGIC
# MAGIC 1. `df.withColumn('Body', regexp_replace(df.body, r'<.*?>', ''))`: This operation creates a new column 'Body' in the DataFrame `df` by replacing all occurrences of HTML tags (e.g., \<p>, \<div>, etc.) in the 'body' column with an empty string. This is achieved using the `regexp_replace` function to perform a regular expression-based replacement.
# MAGIC
# MAGIC 2. `.withColumn("Tags", split(trim(translate(col("Tags"), "<>", " ")), " "))`: This operation creates a new column 'Tags' in the DataFrame `df` by performing a series of transformations:
# MAGIC    - `translate(col("Tags"), "<>", " ")`: This replaces occurrences of '<' and '>' characters in the 'Tags' column with space characters.
# MAGIC    - `trim()`: This removes leading and trailing whitespace from the 'Tags' column.
# MAGIC    - `split(..., " ")`: This splits the 'Tags' column into an array of tags based on space characters.
# MAGIC

# COMMAND ----------

# Formatting the 'Body' and `Tag` columns for machine learning training
df = (df.withColumn('Body', regexp_replace(df.Body, r'<.*?>', '')) # Transforming HTML code to strings
      .withColumn("Tags", split(trim(translate(col("Tags"), "<>", " ")), " ")) # Making a list of the tags
      )
display(df.limit(100))

# COMMAND ----------

# Verify the column types of the DataFrame
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.4. Isolating the "Body" and "Tag" columns. 
# MAGIC <br> 
# MAGIC These are the only one we need.

# COMMAND ----------

TextandTags_df = df.select(col("Body").alias("Text"), col("Tags"))

# Duplicating the text for each possible tag allowing us to have a tag per row instead of a list of tags per text row
TextandTags_df = TextandTags_df.select("Text", explode("Tags").alias("Tags"))

display(TextandTags_df.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.5. Saving the dataframe creating a check point.
# MAGIC

# COMMAND ----------

folder = "/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"
TextandTags_df.write.parquet(f"{folder}TextandTags_df.parquet")

# COMMAND ----------

# Saving the TextandTags_df to memory for repetitive use
TextandTags_df.cache()
TextandTags_df.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##3. Preparing the data for machine learning
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.1. Text Cleaning Preprocessing 
# MAGIC <br>
# MAGIC 1. Replaces any occurrences of URLs, starting with "http" up until the space.
# MAGIC 2. Replaces any non-alphabetic characters in the 'text' column with a space.
# MAGIC 3. Replaces multiple consecutive spaces in the 'text' column with a single space
# MAGIC 4. Converts the 'text' column to lowercase.
# MAGIC 5. Removes leading and trailing whitespace from the 'text' column

# COMMAND ----------

# MAGIC %md
# MAGIC **Reading Saved Dataframe**

# COMMAND ----------

folder = "/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"
TextandTags_df.read.parquet(f"{folder}TextandTags_df.parquet")

# COMMAND ----------

# Preprocessing the data
ml_ready_df = TextandTags_df\
    .withColumn('Text', regexp_replace('Text', r"http\S+", ""))\
    .withColumn('Text', regexp_replace('Text', r"[^a-zA-Z]", " "))\
    .withColumn('Text', regexp_replace('Text', r"\s+", " "))\
    .withColumn('Text', lower('Text')) \
    .withColumn('Text', trim('Text')) 

display(ml_ready_df.limit(50))


# COMMAND ----------

# MAGIC %md
# MAGIC ###3.2. Saving the dataframe creating a check point

# COMMAND ----------

folder = "/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"
ml_ready_df.write.parquet(f"{folder}ml_ready_df.parquet", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC ##4. Machine Learning Model

# COMMAND ----------

# MAGIC %md
# MAGIC Reading saved **ml_ready_df**

# COMMAND ----------

display(dbutils.fs.ls("/tmp/project"))

# COMMAND ----------

display(dbutils.fs.ls("/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"))

# COMMAND ----------

folder = "/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"
ml_ready_df = spark.read.parquet(f"{folder}ml_ready_df.parquet/")

# COMMAND ----------

from pyspark.sql.functions import col, countDistinct
distinct_count = ml_ready_df.agg(countDistinct(col("Tags")).alias("distinct_count"))

# COMMAND ----------

display(distinct_count)

# COMMAND ----------

display(ml_ready_df.limit(10))

# COMMAND ----------

# MAGIC %md 
# MAGIC ###4.1. Feature Transformations
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.1. Tokenizer

# COMMAND ----------

# MAGIC %md
# MAGIC The Tokenizer in PySpark is a feature transformer that converts the input string to lowercase and then splits it by white spaces. It is commonly used to tokenize text data, breaking it into individual terms, usually words. In summary: <br>
# MAGIC 1. Tokenization: The Tokenizer splits the input text into individual terms, typically words, based on white spaces. For example, the sentence "Hi I heard about Spark" would be tokenized into the words ['hi', 'i', 'heard', 'about', 'spark'].
# MAGIC 2. Lowercasing: The input string is converted to lowercase before tokenization.
# MAGIC 3. Output: The output of the Tokenizer transformation is a new column containing the tokenized words.

# COMMAND ----------

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol="Text", outputCol="Tokens")

# Applying the tokenizer to the train data
tokenized = tokenizer.transform(ml_ready_df)

# COMMAND ----------

display(tokenized.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.2. Stopwords Removal

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

stopwords_remover = StopWordsRemover(inputCol="Tokens", outputCol="Filtered")

stopwords = stopwords_remover.transform(tokenized)


# COMMAND ----------

display(stopwords.limit(20))

# COMMAND ----------

# MAGIC %md 
# MAGIC ####4.1.3. CountVectorizer (TF - Term Frequency)

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer

cv = CountVectorizer(vocabSize = 2**16, inputCol="Filtered", outputCol='CV')

# Fitting the CountVectorizer to the training data
cv_model = cv.fit(stopwords)

# Transforming the training data
text_cv = cv_model.transform(stopwords)



# COMMAND ----------

display(text_cv.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.4. TF-IDF Vectorization

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF

idf = IDF(inputCol='CV', outputCol='features', minDocFreq=5)
idf_model = idf.fit(text_cv)
text_idf = idf_model.transform(text_cv)

# COMMAND ----------

display(text_idf.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ###4.2. Label Encoding

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

label_encoder = StringIndexer(inputCol="Tags", outputCol="Label")
le_model = label_encoder.fit(text_idf)
final = le_model.transform(text_idf)

# COMMAND ----------

display(final.limit(20))

# COMMAND ----------

# Save the dataframes for easy retrieval
folder = "/mnt/debdingestiondl/de-bd-project/ML-Training/ModelT_Intermediate_Steps/"

# Saving the 3 tables to mounted data lake
posts.write.parquet(f"{folder}final.parquet")

# COMMAND ----------

final.cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC ###4.3. Model Training

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.3.1. Train Test Split

# COMMAND ----------

train, test = final.randomSplit([0.9, 0.1], seed=202311)

# COMMAND ----------

display(train.limit(10))

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=100)\
  .setLabelCol("Label")
lr_model = lr.fit(train)


# COMMAND ----------

predictions = lr_model.transform(test)

# COMMAND ----------

display(predictions.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ###4.4. Model Evaluation

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol= "prediction", labelCol="Label")
roc_auc = evaluator.evaluate(predictions)
accuracy = predictions.filter(predictions.Label == predictions.prediction).count() / float(predictions.count())

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))

# COMMAND ----------

# MAGIC %md
# MAGIC ##5. Final Model Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ####5.1. Creating the Pipeline

# COMMAND ----------

# Importing all the libraries
from pyspark.sql.functions import split, translate, trim, explode, regexp_replace, col, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Preparing the data
# Step 1: Creating the joined table
df = posts.join(postTypes, posts.PostTypeId == postTypes.Id)
# Step 2: Selecting only Question posts
df = df.filter(col("Type") == "Question")
df = (df.withColumn('Body', regexp_replace(df.Body, r'<.*?>', '')) # Transforming HTML code to strings
      .withColumn("Tags", split(trim(translate(col("Tags"), "<>", " ")), " ")) # Making a list of the tags
      )
# Step 4: Selecting the columns
TextandTags_df = df.select(col("Body").alias("Text"), col("Tags"))
# Step 5: Getting the tags
TextandTags_df = TextandTags_df.select("Text", explode("Tags").alias("Tags"))

# Step 6: Clean the text
ml_ready_df = TextandTags_df\
    .withColumn('Text', regexp_replace('Text', r"http\S+", ""))\
    .withColumn('Text', regexp_replace('Text', r"[^a-zA-Z]", " "))\
    .withColumn('Text', regexp_replace('Text', r"\s+", " "))\
    .withColumn('Text', lower('Text')) \
    .withColumn('Text', trim('Text')) 


# Machine Learning
# Step 1: Train Test Split
train, test = ml_ready_df.randomSplit([0.9, 0.1], seed=202311)

# Step 2: Initializing the transformers
tokenizer = Tokenizer(inputCol="Text", outputCol="Tokens")
stopwords_remover = StopWordsRemover(inputCol="Tokens", outputCol="Filtered")
cv = CountVectorizer(vocabSize = 2**16, inputCol="Filtered", outputCol='CV')
idf = IDF(inputCol='CV', outputCol='features', minDocFreq=5)
label_encoder = StringIndexer(inputCol="Tags", outputCol="Label")
lr = LogisticRegression(maxIter=100, labelCol="Label")

lr = LogisticRegression(maxIter=100, 
                        labelCol="Label",
                        probabilityCol="probability",
                        rawPredictionCol="rawPrediction",
                        predictionCol="prediction"
                        )

# Step 3: Creating the pipeline
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, cv, idf, label_encoder, lr])

# Step 4: Fitting and transforming (predicting) using the pipeline
pipeline_model = pipeline.fit(train)
lr_predictions = pipeline_model.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Improved Pipeline

# COMMAND ----------

# Importing all the libraries
from pyspark.sql.functions import split, translate, trim, explode, regexp_replace, col, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Preparing the data
# Step 1: Creating the joined table
df = posts.join(postTypes, posts.PostTypeId == postTypes.Id)
# Step 2: Selecting only Question posts
df = df.filter(col("Type") == "Question")
df = (df.withColumn('Body', regexp_replace(df.Body, r'<.*?>', '')) # Transforming HTML code to strings
      .withColumn("Tags", split(trim(translate(col("Tags"), "<>", " ")), " ")) # Making a list of the tags
      )
# Step 4: Selecting the columns
TextandTags_df = df.select(col("Body").alias("Text"), col("Tags"))
# Step 5: Getting the tags
TextandTags_df = TextandTags_df.select("Text", explode("Tags").alias("Tags"))

# Step 6: Clean the text
ml_ready_df = TextandTags_df\
    .withColumn('Text', regexp_replace('Text', r"http\S+", ""))\
    .withColumn('Text', regexp_replace('Text', r"[^a-zA-Z]", " "))\
    .withColumn('Text', regexp_replace('Text', r"\s+", " "))\
    .withColumn('Text', lower('Text')) \
    .withColumn('Text', trim('Text')) 


# Machine Learning
# Step 1: Train Test Split
train, test = ml_ready_df.randomSplit([0.9, 0.1], seed=202311)

# Step 2: Initializing the transformers
tokenizer = Tokenizer(inputCol="Text", outputCol="Tokens")
stopwords_remover = StopWordsRemover(inputCol="Tokens", outputCol="Filtered")
cv = CountVectorizer(vocabSize = 2**16, inputCol="Filtered", outputCol='CV')
idf = IDF(inputCol='CV', outputCol='features', minDocFreq=5)
label_encoder = StringIndexer(inputCol="Tags", outputCol="Label")
lr = LogisticRegression(maxIter=100, labelCol="Label")

lr = LogisticRegression(maxIter=100, 
                        labelCol="Label",
                        probabilityCol="probability",
                        rawPredictionCol="rawPrediction",
                        predictionCol="prediction"
                        )

# Step 3: Creating the pipeline
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, cv, idf, label_encoder, lr])

# Step 4: Fitting and transforming (predicting) using the pipeline
pipeline_model = pipeline.fit(train)
lr_predictions = pipeline_model.transform(test)

# COMMAND ----------

# MAGIC %md
# MAGIC ####5.2. Saving the Model to Data Lake 

# COMMAND ----------

# Saving Model
Directory = "/mnt/debdingestiondl/de-bd-project/Model/"

#pipeline_model.save(Directory)

# Saving the model with .write(), instead of the .save(), allows us to include the probability column in the model artefact
pipeline_model.write().option("includeCol", "probability").save(Directory)


# COMMAND ----------

# Extracting the fitted label encoder model from the pipeline model to store it separately
label_encoder_model = pipeline_model.stages[-2]

# Saving the label encode model
Directory = "/mnt/debdingestiondl/de-bd-project/StringIndexer/"
label_encoder_model.save(Directory)

# COMMAND ----------


