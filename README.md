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

## Exploratory Data Analysis

## Modelling

## ML Operations

## Exam Questions

## Generative AI
