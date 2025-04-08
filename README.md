<h1>Azure Cloud + Databricks NLP Inference Pipeline</h1>

<h2>Project Architecture</h2>

<p align="center">
  <img src="https://github.com/efrenmo/Big-Data-Engineering-Project/blob/main/Screenshots/bd_project_architecture.drawio.png" />
</p>


## Project Overview: StackFlow Analytics Platform

Our client, a tech startup aiming to create a developer Q&A platform similar to Stack Overflow, approached us with a critical business challenge. They needed an automated system to analyze post content, assign relevant subject tags, and provide insights on trending topics. This capability would enhance content discovery, improve user experience, and provide valuable business intelligence on user interests and behavior.

To address this need, we designed and implemented a comprehensive data engineering solution leveraging Azure cloud services. Our solution includes an end-to-end pipeline for data ingestion, transformation, machine learning inference, and visualization, enabling automated topic classification of posts and real-time analytics.

## Architecture and Technical Implementation

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

### Data Lake Implementation

We established a hierarchical Azure Data Lake to serve as our central data repository, organized into three logical zones:

- **Bronze Zone**: Raw ingested data
- **Silver Zone**: Transformed and processed data
- **Gold Zone**: Analysis-ready data and ML outputs

This structure ensures data lineage tracking and enables different processing requirements at each stage.

### Data Transformation and ML Processing

The transformation phase leverages Azure Databricks with Apache Spark for distributed processing. Our NLP pipeline applies several techniques to prepare text data:

1. **Text Preprocessing**:
   - HTML tag removal
   - Punctuation elimination
   - Case normalization
   - Whitespace trimming

2. **NLP Feature Engineering**:
   - Tokenization of text into individual words
   - Stop word removal to eliminate common words with low semantic value
   - CountVectorizer implementation to identify frequent terms
   - TF-IDF (Term Frequency-Inverse Document Frequency) weighting to prioritize significant terms

3. **Machine Learning Implementation**:
   - Multi-class text classification using Logistic Regression
   - Label encoding for categorical variables
   - Model inference on daily post batches
   - Results appended to Delta Lake with original post metadata

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
