import boto3
import sagemaker
import sys
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.processing import Processor
from sagemaker.estimator import Estimator
from sagemaker.workflow.parameters import ParameterString
import os

# Fetch AWS region and role from environment or set defaults
region = os.getenv("AWS_REGION", "us-west-2")
role = "AmazonSageMaker-ExecutionRole-20250311T164768"

if not region:
    raise ValueError("AWS_REGION must be set as an environment variable.")

boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session, sagemaker_client=boto_session.client("sagemaker"))

print(f"Using AWS Region: {region}")
print(f"Using SageMaker execution role: {role}")

def get_pipeline():
    # Explicitly pass role as a parameter
    role_param = ParameterString(name="RoleParam", default_value=role)

    # Define processing step (example logic)
    processor = Processor(
        role=role_param,
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/sample-image:latest",
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=sagemaker_session
    )
    step_process = ProcessingStep(name="ProcessingStep", processor=processor, outputs=[])

    # Define training step (example logic)
    estimator = Estimator(
        image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/sample-image:latest",
        role=role_param,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=sagemaker_session
    )
    step_train = TrainingStep(name="TrainingStep", estimator=estimator)

    # Create pipeline and explicitly bind the role parameter
    pipeline = Pipeline(
        name="ark-mlops-jenkins",
        steps=[step_process, step_train],
        sagemaker_session=sagemaker_session,
        parameters=[role_param]
    )
    return pipeline

if __name__ == "__main__":
    pipeline = get_pipeline()

    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        print("Starting pipeline execution...")
        execution = pipeline.start(parameters={"RoleParam": role})
        print(f"Pipeline execution started with ExecutionArn: {execution.arn}")
    else:
        print("Creating SageMaker pipeline...")
        pipeline.create()  # Pass the role explicitly
        print("Pipeline created successfully!")
