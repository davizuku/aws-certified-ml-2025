# AWS Certified Machine Learning Specialty 2025

Notes for the Udemy course: https://www.udemy.com/course/aws-machine-learning/
Additional Resources:
- Course Material: https://www.sundog-education.com/aws-certified-machine-learning-course-materials/

## Data Engineering

- Goal: to have data where it needs to be for training a ML model.

### General knowledge

- Data Partitioning: pattern for speeding up range queries, e.g. date or product
- Durability and Availability
- Data Engineering Pipelines:
    - Real-Time layer:
        Producers
            -> Kinesis Data Streams
                -> Kinesis Data Analytics
                    -> Lambda
                    -> Data Streams -> EC2 <-> SageMaker
                    -> Data Firehose
                        -> S3 -> Redshift
                        -> ElasticSearch
            -> Kinesis Data Firehose
                -> Kinesis Data Analytics
                -> S3
    - Video Layer
        Video Producers
            -> Kinesis Video Streams
                -> Rekognition
                    -> Data Streams -> (...)
                -> EC2
                    -> Data Streams -> (...)
                    <-> SageMaker
    - Batch Layer
        (
        MySQL On-premise -> DMS -> RDS -> Data Pipeline -> S3
        DynamoDB -> Data Pipeline -> S3
        ) -> S3 -> Glue ETL -> S3 <-> Batch for clean up
        * Step Functions for orchestration
        * Glue Data Catalog <- Crawlers
    - Analytics Layer
        S3
            -> EMR (Hadoop / Spark / Hive...)
            -> Redshift (Spectrum) -> QuickSight
            -> Data Catalog -> Athena -> QuickSight

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
- AWS Data Stores
    - Redshift (provisioned): Data Warehousing, SQL analytics, Spectrum, OLAP (online analyitical processing)
    - RDS + Aurora (provisioned): Relational Store, OLTP (online transaction processing)
    - DynamoDB (serverless): NoSQL data store, provision R/W capacity, useful to store ML output.
    - S3: Object storage
    - OpenSearch: ElasticSearch, indexing of data
    - ElastiCache: Caching mechanism
- AWS Data Pipelines Features
    - Destinations S3, RDS, DynamoDB, Redshift
    - Manages task dependencies (orchestration)
    - Runs on EC2 instances
    - Data Pipelines vs. Glue
        - Data Pipelines give more control on underlying infrastructure (EC2, Elastic Map Reduce (EMR), etc.)
    - Data Pipelines is deprecated: https://docs.aws.amazon.com/datapipeline/latest/DeveloperGuide/migration.html
        - Alternatives:
            - Glue
            - Step Functions
            - Managed Workflows for Apache Airflow (MWAA)
- AWS Batch
    - Run jobs as docker images
    - Dynamic provisioning
    - Schedule batch jobs using CloudWatch events
    - Orchestrate batch jobs using Step Functions
    - Batch vs. Glue: Batch should be for non-ETL jobs.
- AWS DMS:
    - Database Migration Service
    - Homogeneous & Heterogeneous migration
    - Continuous Data Replication
    - Needs an EC2 instance for replication
    - DMS vs. Glue: DBM has continous replication, no transformation, only move data.
- AWS Step Functions
    - Design workflows
    - Error Handling & Retry mechanisms
    - History of executions
- AWS DataSync
    - Migrate from on-premises to AWS
    - DataSync agent deployed to On-premises
- MQTT (IoT messaging protocol)
    - A way to transfer lots sensor data to an ML model

## Exploratory Data Analysis

### General Knowledge

- Python libraries:
    - pandas
    - numpy
    - matplotlib
    - seaborn: matplotlib on steroids, pairplot, jointplot
    - scikit_learn
- Data types
    - Numerical
    - Categorical
    - Ordinal: mix of numerical & categorical. Example: star rating
- Data distribution
    - Probability density function (continous)
    - Probability mass function (discrete)
    - Normal -> continuous data
    - Poisson -> discrete data
    - Binomial: number of successes in a sequence of events.
        - Bernoulli: sequence of only one event
- Time Series Analysis
    - Trend
    - Seasonality (periodic occurrences)
    - Noise
    - Additive model (Seasonality + Trends + Noise)
        - Useful when seasonality is constant
    - Multiplicative model (Seasonality * Trends * Noise)
        - Useful when seasonality changes as the trend increases
- Hadoop
    - MapReduce
        - Software to process massive data in parallel
        - Map functions process data
        - Reduce functions combine/aggregate intermediate results into the final format
    - Spark (replaces of MapReduce)
        - In memory caching
        - Directive-acyclic graph for dependency resolution
        - Java, Scala, Python and R
        - Stream processing, ML, ...
        - Modules:
            - Spark Context (driver program)
            - Cluster Managers (Spark, YARN)
            - Executors (cache, tasks)
        - Components:
            - Spark streaming
            - Spark SQL -> similar to python pandas
            - MLLib: distributed and scalable
                - Classification
                - Regression
                - Decision trees
                - Recommendation engine
                - Clustering (K-Means)
                - LDA (topic modeling)
                - ML workflow utilities
                - SVD, PCA, statistics
            - GraphX -> data structure, not charts.
            - Zeppelin: Notebook for spark commands
    - YARN: Yet Another Resource Negotiator
    - HDFS: Hadoop Distributed File System
- Feature Engineering
    - Apply your knowledge of the data to create better features
    - Applied ML is basically feature engineering
    - Curse of Dimensionality: Too many features can be a problem
    - Selecting the relevant features to reduce dimensionality
        - Principal Component Analysis
        - K-Means
    - Missing data imputation
        - Mean replacement
            - Use median if there are outliers
            - Does not usually work well, not accurate, naive
            - Cannot be used on categorical features
        - Dropping
            - Valid approach if small number of missing values, and biased
            - Never would be the *best* approach
        - KNN: Find K "nearest" most similar rows and average their vaues. Bad with categorical
        - Deep Learning: Train a ML model to impute data, hard to achieve, but works great.
        - Regression: MICE (Multiple Imputation by Chained Equations) is state-of-art.
        - Get more data, collecting more real data.
    - Unbalanced data
        - Example: fraud detection
        - Solutions:
            - Oversampling the minority class (random or copying)
                - SMOTE (Synthetic Minority Over-sampling TEchnique): oversampmling using nearest neighbors
            - Undersampling the majority class, not recommended unless "big data" scaling issues.
            - Adjusting thresholds on probability output of the algorithm.
    - Handling outliers
        - An outlier is a point further than X stddev from the mean.
        - Remove outliers only knowing the consequences in the given context.
        - AWS Random Cut Forest algorithm to detect outlier detection
    - Binning
        - Transform numerical data into ordinal/categorical data by bucketing observations together based on ranges
        - Useful to correct errors in measurements
        - Quantile binning, categorized based on the position in the distribution
    - Transforming
        - Applying a function to a feature to make it better suited for training
    - Encoding
        - Transforming data into some new representation: OneHot encoding, Embedding
    - Scaling / Normalization
        - Make the values to be normally distributed around 0.
        - Scale features to be comparable between each other.
    - Shuffling
- TF-IDF
    - Term Frequency: how often a word occurs in a document
    - Inverse Document Frequency: how often a word occurs in an entire set of documents
    - TF / DF = TF * IDF
    - Usually log(IDF)
    - Extension: using n-grams

### AWS Related

- Amazon Athena
    - Features
        - Serverless
        - Query S3 data
        - Formats: CSV, JSON, ORC, Parquet, etc..
    - Use cases
        - Ad-hoc queries of web logs
        - Query before Redshift
        - Integration with Jupyter, Zeppelin, RStudio
        - Integrates with visual tools
        - Athena & Glue:
            - Glue Data Catalog to obtain a schema to be used by Athena
        - NOT FOR:
            - Highly formatted results -> QuickSight
            - ETL -> Glue
    - Cost model
        - 5$ TB scanned
        - Use columnar formats: ORC or Parquet
        - Glue & S3 have their own charges
    - Security
        - IAM
        - Encryption in S3
        - Cross-account using bucket policies
- Amazon QuickSight
    - Data Analysis & Visualization tool
    - Use cases
        - Focuses on business users, not developers
        - Interactive ad-hoc exploration / visualization
        - Dashboards and KPI's
        - ML Insights: Anomaly detection, Forcasting, Auto-narratives
        - NOT FOR:
            - ETL -> Glue
    - Features
        - Serverless
        - Data Sources: Redshift, RDS, Athena, EC2-hosted DBs, S3, IoT, Salesforce, etc.
        - Super-fast, Parallel, In-memory Calculation Engine (SPICE)
        - Columnar storage
        - Limited to 10GB per user
        - Q: answer questions with NLP (training required: dates, topics & datasets)
        - Paginated Reports designed to be printed
        - Dashboards
            - Read only when shared
            - Visual Types: AutoGraph, Bar Chart/Histogram, Line graphs, Scatter plots, Heat Maps, Pie graphs, Tree Maps, Pivot tables, KPIs, Geo Charts, Donuts, Gauge, Word cloud.
    - Security
        - MFA
        - VPC connectivity
        - Row-level & column-level security
        - Private VPC Access
        - User Management: IAM, SAML, Active Directory or email signup
- Elastic MapReduce (EMR) & Hadoop
    - Managed Hadoop framework on EC2 instances
        - Spark, HBase, Presto, Flink, Hive & more
        - EMR Notebooks
    - Use cases
        - Distribute the load to preprocess massive datasets
    - Features
        - Cluster
            - Collection of EC2 instances -> each instance is a Node
            - Node types:
                - Master: m4.large if nodes < 50 else m4.xlarge
                - Core:
                    - m4.large
                    - t2.medium if a lot of i/o
                    - m4.xlarge for improved performance
                    - <see slides>
                 Task: spot instances
            - Core nodes interact with HDFS
            - Task nodes are optional and only execute tasks but do not store data
        - Usage: Transient & Long-Running clusters
        - AWS integration: EC2, VPC, S3, CloudWatch, IAM, CloudTrail, DataPipelines
        - Storage
            - HDFS, distributed, scalable file system -> Ephemeral
            - EMRFS: use S3 as if it were HDFS
            - Local file system
            - EFS for HDFS
        - EMR Notebook:
            - Studios
            - Similar to Zeppelin, but additional AWS features
            - Only available from AWS Console
            - Backed in S3
            - Free to use
    - Cost model
        - Pay per hour
        - Provision new nodes upon failure
        - Add and remove tasks nodes on the fly
        - Resize core nodes in a running cluster
    - Security
        - IAM policies & roles
        - Kerberos
        - SSH
        - Lake Formation config.
        - Apache Ranger for Hadoop / Hive
- SageMaker Ground Truth
    - Having humans to tag data
    - Useful for imputing missing data
    - Big data set
    - It learns from the tagging and only asks for clarification ambiguous cases.
    - Human labelers
        - Mechanical Turk
        - Internal team
        - Profesisonal labeling companies
    - Pre-trained models
        - Rekognition for image recognition
        - Comprehend for text analysis and topic modelling
    Ground Truth *Plus*
        - Hire "AWS Experts" to handle the project for you.
        - Track progress in Plus Project Portal
        - Get labeled data from S3.

## Modelling

### General

#### Neural Networks
- DeepLearning: a neural network with more than one layer.
- Types of neural network:
    - Feedforward -> common classification or regression
    - Convolutional (CNN) -> image classification
    - Recurrent (RNN) -> sequences (time, words, etc.)
- Activation functions
    - Linear activation
    - Binary step function
    - Non-linear functions (backpropagation, multiple layers)
        - Sigmoid / Logistic / TanH
        - Rectified Linear Unit (ReLU)
        - Leaky/Parametric ReLU (small slope for negative numbers)
        - Other ReLU (swish, maxout)
    - Softmax: usually last layer of a classification NN.
    - How to choose one?
        - Multi classification -> softmax on output layer
        - RNN -> TanH
        - Everything else -> ReLU > Leaky ReLU, PReLU, Maxout > Swish (deep networks)
- Convolutional Neural Networks
    - Find patterns in the data
    - Use cases: images, translation, sentence classification, sentiment analysis
    - Keras:
        - Input Data: width * height * color channels
        - Conv2D layer
        - MaxPooling2D
        - Flatten
        - Dropout
        - Dense
        - Dropout
        - Softmax
    - Architectures: (specific arrange of layers)
        - LeNet-5: handwriting recognition
        - AlexNet: image classification
        - GoogLeNet: even deeper
        - ResNet (Residual Network): deeper
- Recurrent Neural Networks
    - Sequence of data: time-series, events, text, music, etc.
    - Past behavior of the model is fed into the current prediction
    - Older data might have lower relevance
    - Topologies
        - Sequence to sequence: predict stock prices
        - Sequence to vector: words in a sentence to sentiment
        - Vector to sequence: captions from an image
        - Encoder -> Decoder: sequence -> vector -> sequence
    - Training usually requires stopping the backpropagation
    - Types of cells:
        - LSTM: long short-term memory
        - GRU: Gated Recurrent Unit, simplification of LSTM
#### Transformers & Modern NLP
- Transformer deep learning architecture
    - self-attention allow processing words in parallel
    - BERT: Bi-directional Encoder Representations from Transformers
    - GPT: Generative Pre-trained Transformer
- Transfer Learning
    - Use pre-trained models
    - Hugging Face has a set of pre-trained models
    - Integrates iwth Sagemaker
    - Approaches
        - Use it as-is
        - Fine-tune
            - Format input data as it was when the model was trained originally
            - Start training the model with low learning rate.
        - Add new trainable layers on top of the existing model
        - Retrain from scratch using the architecture
            - beware of data and money!
#### Tuning Neural Networks (Hyperparameters)
- Learning Rate
    - Training is based on gradient descent on a cost function
    - Learning Rate sets up the rythm.
        - High LR -> overshoot correct solution
        - Low LR -> too slow
- Batch size
    - How many training samples are used within each batch of each epoch
    - Small batch sizes can skip local minimum
    - Big batch sizes can get stuck into local minimum
- Regularization
    - Prevent overfitting
    - Techniques
        - Simpler model
        - Dropout
        - Early stopping
        - L1: sum of weights -> feature selection -> Sparse output
        - L2: sum of weights^2 -> all features considered, just weighted -> Dense output
#### Evaluation and debugging
- Vanishing Gradient Problem
    - When gradient approaches zero (near the local minimum) computational problems arise with very low numbers.
    - Fixes:
        - Multi-level hierarchy: train levels separately
        - Specific architectures: LSTM, ResNet
        - Better activation function: ReLU is a good choice (because of its 45º straight line)
    - Gradient checking is a debugging/diagnostic tool
- Confusion Matrix
    - Accuracy might not tell the whole story, usually with anomalies.
    - True Positives: Actual YES, Predicted YES (✓)
    - False Positives: Actual NO, Predicted YES (x)
    - False Negatives: Actual YES, Predicted NO (x)
    - True Negative: Actual NO, Predicted NO (✓)
- Precision, Recall, F1, AUC
    - Recall = Sensitivity = Completeness = TP / (TP + FN)
        - Used mainly in fraud detection
    - Precision = TP / (TP + FP)
        - Used in drug testing, medical screening
    - Specificity = TN / (TN + FP)
    - F1 score = 2 * (Precision * Recall) / (Precision + Recall)
        - Used when we care about both Precision and Recall
    - RMSE: Root Mean Squared Errors
    - ROC Curve: Receiver Operating Characterisctic Curve
        - The more "bent" towards the upper-left corner, the better
    - AUC: Area under the ROC Curve
        - AUC of 0.5 is a useless model
        - AUC of 1.0 is a perfect model
    - P-R Curve: Precision/Recall curve
        - The higher the area under the curve, the better.
        - Good for information retrieval and large number of documents
#### Ensemble Methods
- Take multiple models and let them vote for the final result
- Bagging
    - Generate N new training sets by random sampling with replacement
    - Avoids overfitting
    - Easier to parallelize
- Boosting
    - Sequentially train multiple models
    - Some data points will repeat
    - Output from previous models is considered for next models
    - XGBoost

### AWS Specific

#### Amazon SageMaker
- Designed to handle the full cycle of ML
    - Fetch, clean, prepare data
        - Data in S3
        - ECR contains a docker image for training
        - Save model in S3
        - ECR contains a docker image for inference
    - Train and evaluate model
    - Deploy model, evaluate results in production
        - Persistent endpoint: SageMaker create endpoints allowing access to the inference docker.
        - Batch transform
        - SageMaker Neo for edge devices
        - Shadow Testing to catch errors comparing new models against production ones
- SageMaker Notebooks
    - Access S3
    - Scikit_learn, Spark, Tensorflow
    - Start training jobs
    - Deploy models
    - etc.
- Built-In Algorithms
    |Name|Description|Use case|Input format|Recommended Infrastructure|Important hyper params|
    |----|-----------|--------|------------|--------------------------|----------------------|
    |Linear Learner||Regression & Classification|<ul><li>RecordIO (best) or CSV (first col = label)</li><li>use Pipe or File</li><li>Normalized and shuffled</li></ul>|Single or multi-instance CPU or GPU|<ul><li>balance_multiclass_weights</li><li>learning_rate, mini_batch_size </li><li>L1, Wd (weight decay = L2)</li><li>target_precision</li><li>target_recall</li>|
    |XGBoost|<ul><li>Boosted group of decision trees </li><li>New models to correct errors of previous trees</li></ul>|Mainly for classification, but also for regression|CSV, RecordIO-protobuf, and Parquet|It is memory bound; recommended use m5. After XGBoost 1.5+ support for Distributed GPU training|Use `import sagemaker.xgboost`:<ul></ul><li>subsample, eta: to prevent overfitting </li><li>gamma: min loss reduction to create a new tree</li><li>alpha: L1 regularization </li><li>lambda: L2 regularization</li><li>eval_metric: 'auc', 'error', 'rmse', ... </li><li>scale_pos_weight: deal with unbalanced data </li><li>max_depth: too high -> overfitting</li><li>tree_method: gpu_hist (when GPU available)</li></ul>|
    |Seq2Seq|Sequence of tokens -> sequence of tokens|Machine translation|RecordIO-Protobuf<ul><li>Tokens must be integers</li><li>Tokenized text files</li><li>Training data, validation data, vocabulary files</li></ul>|GPU instance types. Only single instance, P3|<ul><li>batch_size</li><li>optimizer_type</li><li>learning_rate</li><li>num_layers_encoder</li><li>num_layers_decoder</li><li>Optimize on BLEU socre, perplexity (cross-entropy)</li></ul>|
    |DeepAR|<ul><li>Several time series support, finds frequencies and seasonality</li><li>Use the entire time series both for training and inference</li><li>Don't use more than 400 prediction</li></ul>|Forecasting one-dimensional time series data (RNN)|<ul><li>JSON lines (gzip or parquet)</li><li>each record: `start`<timestamp>, `target`:<value>, `dynamic_feat`, `cat`</li></ul>|<ul><li>CPU or GPU instances </li><li>Single or multi machine</li><li>Start with CPU instances ml.c4.2xlarge</li><li>Only move to GPU if necessary (large models or mini-batch sizes)</li><li>CPU-only for inference</li></ul>|<ul><li>context_length</li><li>epochs</li><li>mini_batch_size</li><li>learning_rate</li><li>num_cells</li></ul>|
    |BlazingText|Not LLM|Text classification (supervised), Word2vec|Text Classification<ul><li>One sentence por line, staring with `__label__ <label>`</li><li>Augmented manifest text format {"source": "text", "label": <label>}</li></ul><br/>Word2Vec: Text file with one sentence per line|Text classification:<ul><li>C5 if less than 2GB training data</li><li>P2 or P3 for larger data sets</li></ul><br/>Word2vec:<ul><li>cbow & skipgram: ml.p3.2xlarge</li><li>batch_skipgram: can use multiple CPUs</li></ul>| Text Classification<ul><li>epochs</li><li>learning_rate</li><li>word_ngrams</li><li>vector_dim</li></ul><br/>Word2vec<ul><li>mode: batch_skipgram, skipgram, cbow</li><li>learning_rate</li><li>window_size</li><li>vector_dim</li><li>negative_samples</li></ul>|
    |Object2Vec|<ul><li>Embedding transformation for more general objects than BlazingText</li><li>Train with two input channels -> two encoders -> comparator -> label</li></ul>|<ul><li>Nearest neighbors</li><li>Visualize clusters</li><li>Genre prediction</li><li>Recommendations (similar items or users)</li></ul>|<ul><li>Tokenized of integers</li><li>Sequence of pairs of tokens</li><li>Example: `{"label": 0, "in0": [...], "in1": [...]}</li></ul>|<ul><li>Single machine CPU, GPU or multi-GPU</li><li>Start m5.2xlarge or p2.xlarge</li><li>GPU options: p2, p3, g4dn, g5</li><li>Inference use ml.p3.2xlarge with INFERENCE_PREFERRED_MODE env var.</li></ul>|<ul><li>dropout</li><li>early_stopping</li><li>epochs</li><li>learning_rate</li><li>batch_size</li><li>layers</li><li>activation_function</li><li>optimizer</li><li>weight_decay (L2 reg)</li><li>enc1_network: hcnn, bilstm, pooled_embedding</li><li>enc2_network: hcnn, bilstm, pooled_embedding</li></ul>|
    |ObjectDetection|<ul><li>Variants: MXNet and Tensorflow</li><li>Allows retraining</li></ul>|Identify all objects in an image with bounding boxes|<ul><li>RecordIO</li><li>image format (jpg or png) + json file for annotation `{"top": <int>, "left": <int>, "width": <int>, "height": <int>}`</li></ul>|<ul><li>Single and multi instances</li><li>GPU & multi-GPU</li><li>ml.p2 or p3 instances</li><li>Instance: m5, p2, p4, g4dn</li></ul>|<ul><li>mini_batch_size</li><li>learning_rate</li><li>optimizer: sgd, adam, rmsprop, adadelta</li></ul>|
    |Image Classification|<ul><li>Variant: MXNet and Tensorflow</li><li>MXNet: full training & transfer learning. default size 3-channel 224x224</li><li>Tensorflow: various hub models (MobileNet, Inception, ...). Top classification layer available for fine tunning.</li></ul>|Assign one or more labels to an image without telling where.|<ul><li>Single & multi-instances</li><li>GPU and multi-GPU</li><li>Training: p2, p3, g4dn, g5</li><li>Inference: CPU or GPU (m5, p2, p3, g4dn, g5)</li></ul>|<ul><li>batch_size</li><li>learning_rate</li><li>optimizer</li><li>weight_decay</li><li>beta1</li><li>beta2</li><li>eps</li><li>gamma</li></ul>|
    |Semantic Segmentation|<ul><li>Pixel-level object classification using a _segmentation mask_</li><li>Built on MXNet Gluon and GluonCV</li><li>Algorithms: FCN, PSP, and DeepLabV3</li><li>Backbones: ResNet50, ResNet101</li><li>Full training and train from scratch</li></ul>| self-driving vehicles, imaging diagnosis, robot sensing.|<ul><li>JPG Images and PNG annotations</li><li>label maps</li><li>Augmented manifest image format for Pipe mode</li></ul>|<ul><li>Training: single machine only GPU (p2, p3, g4dn, g5)</li><li>Inference: CPU (c5 or m5) or GPU (p3 or g4dn)</li></ul>|<ul><li>epochs</li><li>learning_rate</li><li>batch_size</li><li>optimizer</li><li>algorithm</li><li>backbone</li></ul>|
    |Random Cut Forest|<ul><li>Creates a forest of trees is a partition of the training data. </li> <li>If a new point causes a lot of changes in the decision tree, it seems suspicious</li> <li>It is used in Kinesis Analytics, working on streaming data.</li></ul>|<ul> <li>Anomaly detection unsupervised. </li> <li>Spikes in time series, breaks in periodicity, etc. </li></ul>|<ul><li>CSV or RecordIO</li> <li>File or Pipe mode </li> <li>(optional) Test channel for computing accuracy, precision, recall, and F1 on labeled data</li></ul>|<ul><li>Only CPU</li> <li>Training: m4, c4 or c5</li> <li>Inference: c5.xl</li></ul>|<ul><li>num_trees</li> <li>num_samples_per_tree: 1/num_samples_per_tree approximates the ratio of anomalous to normal data</li></ul>|
    |Neural Topic Model|<ul> <li>What is a document about? </li><li>Unsupervised algorithm</li><li>How many topics we want? </li></ul>|<ul><li>Organize documents into topics</li><li>Classify or summarize documents based on topics</li></ul>|<ul><li>Four data channels: train (required), validation, test, auxiliary</li><li>File or Pipe mode</li><li>RecordIO-protobuf or CSV</li><li>words must be tokenized into integers</li><li>vocabulary must be specified in "auxiliary" channel</li></ul>|<ul><li>Training: CPU or GPU</li><li>Inference: CPU</li></ul>|<ul><li>mini_batch_size</li><li>learning_rate</li><li>num_topics</li></ul>|
    |Latent Dirichlet Allocation (LDA)|<ul><li>Unsupervised</li><li>Topic modelling algorithm not deep learning</li><li>Cheaper than Neural Topic Model</li></ul>|<ul><li>Cluster things other than text: customer purchases, harmonic analysis, etc.</li></ul>|<ul><li>Train channel, optional test channel</li><li>RecordIO-protobuf or CSV</li><li>counts for every word in document (vocabulary)</li><li>Pipe mode only with RecordIO</li></ul>|<ul><li>Training: single instance, only CPU</li><li>Inference: CPU</li></ul>|<ul><li>num_topics</li><li>alpha0: initial guess for concentration parameter.</li></ul>|

#### Where to run and train deep models
- EMR supports Apache MXNet and GPU
- Appropriate EC2 types for deep learning: P3, P2, G3, G5g, P4d - A100 "UltraClusters"
- Generative AI: Trn1, Trn1n instances (high bandwidth between nodes),
- Inference: Inf2


## ML Operations

## Exam Questions

## Generative AI
