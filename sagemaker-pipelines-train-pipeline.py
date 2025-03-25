import sagemaker
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.workflow.pipeline_context import PipelineSession
import boto3

# Initialize SageMaker session and role
region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# SageMaker XGBoost Image URI (Example)
image_uri = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.0-1-cpu-py3"

# Define Estimator for training step
xgb_estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://your-s3-bucket/output',
    sagemaker_session=sagemaker_session,
    hyperparameters={
        'max_depth': 5,
        'eta': 0.2,
        'objective': 'binary:logistic',
        'num_round': 100,
        'eval_metric': 'auc'
    }
)

# Define training and validation datasets from S3
train_data = TrainingInput(f's3://your-s3-bucket/train/', content_type='csv')
val_data = TrainingInput(f's3://your-s3-bucket/val/', content_type='csv')

# Define the TrainingStep
train_step = TrainingStep(
    name="TrainModelStep",
    estimator=xgb_estimator,
    inputs={'train': train_data, 'validation': val_data}
)

# Define the Pipeline
pipeline = Pipeline(
    name="MySageMakerPipeline",
    steps=[train_step]
)

# Create and start the pipeline execution
if __name__ == "__main__":
    # Create or update the pipeline
    pipeline.upsert(role_arn=role)

    # Start the pipeline execution
    execution = pipeline.start()
    execution.wait()
    print("Pipeline execution completed.")
