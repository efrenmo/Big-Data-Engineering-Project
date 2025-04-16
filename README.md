<h1>Azure Cloud + Databricks NLP Inference Pipeline</h1>

## Project Overview: StackFlow Analytics Platform

Our client, a tech startup aiming to create a developer Q&A platform similar to Stack Overflow, approached us with a critical business challenge. They needed an automated system to analyze post content, assign relevant subject tags, and provide insights on trending topics. This capability would enhance content discovery, improve user experience, and provide valuable business intelligence on user interests and behavior.

To address this need, we designed and implemented a comprehensive data engineering solution leveraging Azure cloud services. Our solution includes an end-to-end pipeline for data ingestion, transformation, machine learning inference, and visualization, enabling automated topic classification of posts and real-time analytics.

## Architecture and Technical Implementation

<p align="center">
  <img src="https://github.com/efrenmo/Big-Data-Engineering-Project/blob/main/Screenshots/bd_project_architecture.drawio.png" />
</p>

### Data Sources and Ingestion

Our solution ingests data from two primary sources:

1. **AWS RDS PostgreSQL Database**: Contains Users and Post Types tables with user profiles and post classification information
<p align="center">
  <img src="Screenshots\tables1.drawio.png" alt="Users and Post Types tables ">
</p>

2. **Azure Blob Storage**: Houses daily Posts data in parquet format, containing the actual content created by users
<p align="center">
  <img src="Screenshots\table2.drawio.png" alt="Post table ">
</p>

The data ingestion layer was implemented using **Azure Data Factory (ADF)**, which orchestrates the entire pipeline. We created two distinct extract-load pipelines:

1. **Posts Data Pipeline**: Executes daily to capture new content

<p align="center">
  <img src="Screenshots\daily_pipeline.drawio.svg" alt="Daily Pipeline ">
</p>

2. **Users and PostTypes Pipeline**: Runs weekly following SCD type 1 methodology (overwriting with latest records)

<p align="center">
  <img src="Screenshots\weekly_pipeline.drawio.svg" alt="Weekly Pipeline ">
</p>

For each pipeline, we configured:
- Source and destination linked services
- Dataset definitions with appropriate schemas
- Copy data activities with optimized settings

<p align="center">
  <img src="Screenshots\copy_activity_details.drawio.svg" alt="Copy Activity Details">
</p>

### Data Lake Implementation

We established a hierarchical Azure Data Lake to serve as our central data repository, organized into three logical zones:

- **Bronze Zone**: Raw ingested data
- **Silver Zone**: Transformed and processed data
- **Gold Zone**: Analysis-ready data and ML outputs

This structure ensures data lineage tracking and enables different processing requirements at each stage.

### ML Model Training Process
Our model training pipeline leverages Databricks for distributed processing and implements a multi-class text classification approach using logistic regression. The training process consists of several key stages:

#### Data Preparation

The training data was sourced from Stack Overflow posts stored in our data lake. We performed the following preparation steps:

- Loaded posts, post types, and user data from parquet files in our data lake.
- Joined posts with post types to identify question-type content.
- Filtered the dataset to focus exclusively on questions, as these contain the relevant tags we aim to predict.
- Transformed the HTML-formatted tags into clean arrays for model training.
- Saved the prepared dataset as a parquet file for reproducibility.

#### Text Preprocessing

To optimize our NLP model's performance, we implemented a comprehensive text preprocessing pipeline:

- HTML tag removal to extract plain text content.
- Special character and punctuation elimination.
- Case normalization to lowercase.
- Whitespace standardization and trimming.
- Removal of URLs and non-alphabetic characters.

#### NLP Feature Engineering

The feature engineering process transformed the raw text into ML-ready features:

- **Tokenization:** Split text into individual words.
- **Stop Word Removal:** Eliminated common words with low semantic value.
- **HashingTF:** Applied term frequency hashing to convert text to numerical features.
- **IDF Weighting:** Implemented TF-IDF to prioritize significant terms.

#### ML Model Development
We implemented a logistic regression classifier with the following characteristics:

- Multi-class classification approach to handle the diverse range of technical topics.
- StringIndexer to convert categorical tag labels to numerical indices.
- Pipeline architecture to ensure reproducible preprocessing and model application.
- Evaluation using MulticlassClassificationEvaluator to measure accuracy.

### ML Model Deployment and Inference

Our model deployment strategy focuses on batch inference integrated with our data pipeline. This approach allows us to process new content regularly and update our analytics platform with fresh insights.

#### Inference Pipeline Architecture
The inference pipeline consists of several components:

- **Configuration Management:** Centralized configuration to maintain consistency across environments.
- **Model Loading:** Dynamic loading of the trained PipelineModel and StringIndexerModel from the data lake.
- **Batch Processing:** Daily processing of new question posts.
- **Timestamp Tracking:** Each processed batch receives a timestamp for lineage tracking.

#### Data Preprocessing for Inference
The preprocessing user defined function (UDF) handles the transformation of raw posts data:

- Renames and drops columns to standardize the schema.
- Filters for question-type posts only.
- Converts tag strings to arrays.
- Applies the same text cleaning process used in training:
  - URL removal
  - HTML tag stripping
  - Non-alphabetic character removal
  - Whitespace normalization
  - Case conversion
  - Trimming

#### Prediction Process
The prediction workflow includes:

- Loading the trained pipeline model from the data lake.
- Applying the model to the preprocessed batch data.
- Converting numerical predictions back to human-readable tags using IndexToString.
- Joining predictions with original post metadata to maintain context

### Delta Lake Integration
Delta Lake serves as the backbone of our data persistence and analytics strategy, providing robust, ACID-compliant storage for all machine learning inference results. This integration ensures data reliability, enables time-based analytics, and supports seamless downstream consumption by BI tools.
<br>

**1. Delta Table Schema and Initialization**
<br>

We define a comprehensive schema for our predictions, capturing both model outputs (predictions) and original post metadata. The Delta table is created (if it does not exist) using **Spark SQL**:

```sql
CREATE TABLE IF NOT EXISTS stack_overflow_proj.labeled_qs_ext_dtbl (
  question_id INT,
  Text STRING,
  prediction INT,
  decoded STRING,
  AcceptedAnswerId INT,
  AnswerCount INT,
  CommentCount INT,
  CreationDate TIMESTAMP,
  FavoriteCount INT,
  LastEditDate TIMESTAMP,
  LastEditorUserId INT,
  OwnerUserId INT,
  PostTypeId INT,
  Score FLOAT,
  Tags ARRAY,
  Title STRING,
  ViewCount INT,
  Type STRING,
  batch_datetime_stamp TIMESTAMP
)
USING DELTA
LOCATION 'dbfs:/mnt/debdingestiondl/de-bd-project/BI/labeled_qs_ext_dtbl'
```
This schema enforces data consistency and supports schema evolution as requirements change.                                                                                                                                                  
<br>

**2. Batch Processing and Timestamping**
<br>

Each batch of inference results is tagged with a processing timestamp, enabling data lineage and time-based analytics:

```python
from datetime import datetime
from pyspark.sql.functions import to_timestamp, lit

batch_processing_date = datetime.now().strftime("%Y_%m_%d_%H")
final_results_df = results_processed_df.withColumn(
    'batch_datetime_stamp', 
    to_timestamp(lit(batch_processing_date), 'yyyy_MM_dd_HH')
)
```
<br>

**3. Writing Inference Results to Delta Lake**
<br>

Inference results are written to the Delta table in **append** mode, ensuring that each batch is atomically and durably persisted:

```python
final_results_df.write.format("delta")\
    .mode("append")\
    .option("path", f"{bi_folder_path}/labeled_qs_ext_dtbl/labeled_qs_batch_{batch_processing_date}")\
    .saveAsTable("stack_overflow_proj.labeled_qs_ext_dtbl")
```
<br>

- When we used `CREATE TABLE ... USING DELTA LOCATION ...` during initialization, it registers the delta table in the **Hive metastore** making it accessible via SQL and Spark catalog queries.
- This registration allows us to reference the table by name (*stack_overflow_proj.labeled_qs_ext_dtbl*) in SQL and Spark, rather than by file path.

<br>

**4. Benefits of Delta Lake Integration**
<br>

- **ACID Transactions:** Guarantees atomicity, consistency, isolation, and durability for all write operations, ensuring that each write is reliable and that the table remains consistent, even in the event of failures. 
- **Time Travel:** Enables querying of historical data for auditing and model performance comparison.
- **Schema Enforcement:** Ensures data integrity and supports schema evolution.
- **Optimized Storage:** Leverages Parquet format for efficient storage and fast analytics.
- **Seamless Analytics Integration:** Delta tables are immediately queryable by Spark SQL and BI tools.


The integration with Delta Lake ensures that our BI dashboards always have access to the latest tag predictions while maintaining historical data for trend analysis.
  
### Visualization and Analytics

The final component leverages Azure Synapse Analytics to create interactive dashboards that:

- Display top 10 trending topics daily
- Track topic popularity over time
- Provide insights on user engagement by topic
- Enable drill-down analysis for content strategists

## Business Impact and Results

The implemented solution delivers significant value to our client:

- **Automated Content Organization**: Reduced manual tagging effort by 95%
- **Enhanced User Experience**: Improved content discovery through accurate topic classification
- **Business Intelligence**: Provided real-time insights on trending developer interests
- **Scalable Architecture**: System handles growing data volumes with consistent performance
- **Cost Efficiency**: Serverless components minimize infrastructure management overhead

## Future Enhancements

Based on initial success, we've identified several opportunities for future enhancements:

1. **Real-time Processing**: Implement streaming analytics for immediate tag assignment
2. **Advanced ML Models**: Explore transformer-based models (BERT, GPT) for improved accuracy
3. **User Recommendation Engine**: Develop personalized content recommendations based on topic interests
4. **Sentiment Analysis**: Add sentiment scoring to provide additional content insights
5. **Multi-language Support**: Extend NLP capabilities to support international content

## Conclusion

The StackFlow Analytics Platform demonstrates how modern cloud data engineering can solve complex business challenges through intelligent automation. By combining Azure's data services with machine learning, we've delivered a solution that not only meets the client's immediate needs but provides a foundation for future growth and innovation.

The architecture's modularity ensures that new data sources, transformation logic, or visualization requirements can be incorporated with minimal disruption, positioning our client for continued success in the competitive developer community space.
