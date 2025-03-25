import boto3
import sagemaker

# Fetch AWS region from the environment variable or set a default region
import os
region = os.getenv("AWS_REGION", "us-west-2")  # This will use the AWS_REGION if set, or default to 'us-west-2'

# Create a Boto3 session with the region
boto_session = boto3.Session(region_name=region)

# Create a SageMaker session using the Boto3 session
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# Get the execution role (This assumes you're running on an environment like SageMaker Studio or EC2 with an IAM role attached)
role = "AmazonSageMaker-ExecutionRole-20250311T164768"

print(f"Using AWS Region: {region}")
