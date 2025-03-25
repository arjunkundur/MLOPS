import boto3
import sagemaker
import sys
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import Processor
from sagemaker.estimator import Estimator

# Fetch AWS region from the environment variable or set a default region
import os
region = os.getenv("AWS_REGION", "us-west-2")  # This will use the AWS_REGION if set, or default to 'us-west-2'

# Create a Boto3 session with the region
boto_session = boto3.Session(region_name=region)

# Create a SageMaker session using the Boto3 session
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# Define SageMaker execution role (replace it with your correct role ARN if needed)
role = "AmazonSageMaker-ExecutionRole-20250311T164768"

print(f"Using AWS Region: {region}")
print(f"Using SageMaker execution role: {role}")

# Define a simple SageMaker pipeline
def get_pipeline():
    # Dummy Processor step (replace with actual data processing logic)
    processor = Processor(role=role, image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/sample-image:latest", instance_count=1, instance_type="ml.m5.large")
    step_process = ProcessingStep(name="ProcessingStep", processor=processor, outputs=[])

    # Dummy Training step (replace with actual model training logic)
    estimator = Estimator(image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/sample-image:latest", role=role, instance_count=1, instance_type="ml.m5.large")
    step_train = TrainingStep(name="TrainingStep", estimator=estimator)

    # Create and return the pipeline
    pipeline = Pipeline(
        name="ark-mlops-jenkins",
        steps=[step_process, step_train]
    )
    return pipeline

if __name__ == "__main__":
    pipeline = get_pipeline()

    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        print("Starting pipeline execution...")
        execution = pipeline.start()  # Start pipeline execution
        print(f"Pipeline execution started with ExecutionArn: {execution.arn}")
    else:
        print("Creating SageMaker pipeline...")
        pipeline.create()  # Create (register) the pipeline in SageMaker
        print("Pipeline created successfully!")
