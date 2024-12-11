# AWS Certified Machine Learning Specialty 2025

Notes for the Udemy course: https://www.udemy.com/course/aws-machine-learning/
Additional Resources:
- Course Material: https://www.sundog-education.com/aws-certified-machine-learning-course-materials/

## Data Engineering

- Goal: to have data where it needs to be for training a ML model.

### General knowledge

- Data Partitioning: pattern for speeding up range queries, e.g. date or product
- Durability and Availability

### AWS Related

- AWS Budgets
- Amazon S3
    - General
        - Object storage in buckets
        - Globally unique name
        - Key is the full path
        - Partitioning
        - Max object size is 5TB
        - Object tags <key, value>
    - ML-related
        - Backbone for ML services, e.g. SageMaker
        - Data Lake
        - Centralized architecture
        - Any file format: CSV, JSON, Parquet, ORC, Avro, Protobuf
    - Bucket policies
    - Encryption
        - SSE-S3:
            - key is handled, managed, and owned by AWS S3
            - Set header: `"x-amz-server-side-encryption":"AES256"`
        - SSE-KMS:
            - keys managed by AWS KMS
            - User control + audit key usage using Cloud Trail
            - Set header: `"x-amz-server-side-encryption":"aws:kms:<arn>"`
            - Limitations on KMS service, quota per second
        - SSE-C:
            - Keys provided by customers, via HTTPS, in headers
            - Amazon S3 will not store the key
        - Client-Side Encryption
            - Clients encrypt/decrypt data outside S3
        - In transit: SSL/TLS
            - S3 exposes two endpoints HTTP & HTTPS
            - Force HTTPS using Bucket policy: `"aws:SecureTransport": "false"`
    - Force encryption using a Bucket policies, examples:
        - `"s3:x-amz-server-side-encryption": "aws:kms"`
        - `"s3:x-amz-server-size-encryption-customer-algorithm": "true"`
    - VPC Endpoint GW:
        - EC2 instances need to go through an Internet GW (public access)
        - EC2 instances need to go through an VPC Endpoint GW (private access)
            - Bucket policy: `AWS:SourceVpce` or `AWS:SourceVpc`
- AWS Kinesis
    - Streaming service alternative to Apache Kafka
    - Kinesis Streams (REAL TIME)
        - Ingesting data
        - Use cases:
            - Streams are divided into Shards / Partitions
            - Not for petabyte analysis
        - Features
            - Capacity Modes: Provisioned & On-demand
            - Producer: write 1MB/s or 1000 messages/s PER SHARD
            - Consumer: read 2MB/s or 5 API messages/s PER SHARD
            - Real-time latency: 70-200 ms
            - Data Storage for 1 to 365 days, replay capability, multi consumers
    - Kinesis Firehose (DELIVERY / INGESTION)
        - Moving (massive) data to S3 or Redshift
        - Use cases:
            - Can read from Kinesis Streams, CloudWatch or AWS IoT
            - Can use lambda to transform data
            - Batch writes into a Destination
            - Destinations:
                - AWS: S3, Redshift (via S3), ElasticSearch
                - 3rd Party: datadog, mongodb, new relic, splunk
                - Custom destinations via HTTP endpoint
            - Failed data to S3 backup bucket
        - Features:
            - Fully managed service: near real time (buffered)
            - Data Conversions CSV/JSON -> Parquet/ORC
            - Pay for data going through it
    - Kinesis Analytics
        - Real-time ETL / ML algorithms on streams
        - SQL on stream data
        - Use cases:
            - Streaming ETL: select columns, simple transformations
            - Continuous metric generation: live leader board
            - Responsive analytics: look for criteria and build alerting
        - Features
            - Pay for resources consumed (not cheap)
            - Serverless
            - IAM permissions for accessing sources and destinations
            - SQL or Flink to write the computation
            - Schema discovery
            - Lambda for preprocessing
            - Blue print
            - AWS Lambda can be also a destination
            - Managed Service for Apache Flink
                - Bring your own Flink App (Flink Sources + Flink Sinks)
        - ML integration
            - RANDOM_CUT_FOREST: anomaly detection on numeric columns based on recent history
            - HOTSPOTS: locate and return information about relatively dense regions in your data
    - Kinesis Video Streams
        - Sending video
        - Producers: AWS DeepLens, RTSP camera, etc.
            - Convention: One producer per video stream
        - Consumers:
            - Build your own model (TF, MXNet)
            - AWS SageMaker
            - Amazon Rekognition Video
        - Features:
            - Video Playback
            - Retention 1 to 10 years
        - Use cases
            - Consume stream in real-time (Video Stream)
            - Check point stream (with DynamoDB) in order to resume operation upon abort
            - Send decoded frames for ML-based inference (SageMaker)
            - Publish inference results (Data Streams)
            - Notifications (Lambda)
- Glue
    - Data Catalog
        - Metadata repository for all schemas in your account
        - Automated Schema Inferece
        - Schemas versioned
    - Crawlers
        - Iterate data to infer schemas and partitions
        - Stores: S3, Redshift, RDS
        - Partition is important to be though in advance
    - Glue ETL: Extract+Transform+Load
        - Jobs are run on a serverless Spark platform
            - Source -> Transforms -> Targets
            - Role needed (overdimensioned):
                - AmazonS3FullAccess + AWSGlueServiceRole
        - Scheduler (cron)
        - Trigger (events)
        - Transformations:
            - Bundled (DropFields, Filter, Join, Map)
            - ML (FindMatches ML): identify duplicate or matching records
            - Apache Spark transformations (e.g. K-Means)
        - Conversions between CSV, JSON, Avro, Parquet, ORC, XML
    - Glue Data Brew
        - Clean and normalize data without writing any code
        - Data source S3, Redshift, Aurora, Glue Data Catalog...
        - +250 ready-made transformation (filters, conversion, invalid values, etc.)
        - All actions are recorded into a "Recipe"
        - Then a job runs the "Recipe"
    - Athena
        - Use SQL to search over multiple S3 files, that were previously crawled by Glue.

## Exploratory Data Analysis

## Modelling

## ML Operations

## Exam Questions

## Generative AI
