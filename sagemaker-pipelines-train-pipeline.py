import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import Processor
from sagemaker.estimator import Estimator
import os

# Fetch AWS region and role from environment or fallback values
region = os.getenv("AWS_REGION", "us-west-2")
role = "arn:aws:iam::123456789012:role/AmazonSageMaker-ExecutionRole-20250311T164768"

if not region:
    raise ValueError("AWS_REGION must be set.")

print(f"Using AWS Region: {region}")
print(f"Using SageMaker execution role: {role}")

boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session, sagemaker_client=boto_session.client("sagemaker"))

# Print session details to debug
print(f"SageMaker session: {sagemaker_session}")
print(f"Default bucket: {sagemaker_session.default_bucket()}")

def get_pipeline():
    # Create processing step with role hardcoded
    processor = Processor(
        role=role,
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/sample-image:latest",
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=sagemaker_session
    )
    step_process = ProcessingStep(name="ProcessingStep", processor=processor, outputs=[])

    # Create training step with role hardcoded
    estimator = Estimator(
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/sample-image:latest",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=sagemaker_session
    )
    step_train = TrainingStep(name="TrainingStep", estimator=estimator)

    # Create pipeline with role hardcoded
    pipeline = Pipeline(
        name="ark-mlops-jenkins",
        steps=[step_process, step_train],
        sagemaker_session=sagemaker_session
    )
    return pipeline

if __name__ == "__main__":
    pipeline = get_pipeline()

    print("Creating SageMaker pipeline...")
    try:
        pipeline.create()  # Attempt pipeline creation
        print("Pipeline created successfully!")
    except Exception as e:
        print(f"Error during pipeline creation: {str(e)}")
