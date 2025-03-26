import os
import boto3
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.session import Session

def main():
    # Configuration
    project = "my-sagemaker-aws-jenkins-project1"
    role = "arn:aws:iam::975050337104:role/service-role/AmazonSageMaker-ExecutionRole-20250311T162664"
    region = 'ap-south-1'
    
    # Get absolute path to current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Debug: Verify paths
    print(f"Current directory: {current_dir}")
    print(f"train.py exists: {os.path.exists(os.path.join(current_dir, 'train.py'))}")
    
    # Initialize session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(boto_session=boto_session)
    
    # Create estimator with explicit source_dir
    estimator = XGBoost(
        entry_point='train.py',
        source_dir=current_dir,  # Critical: Tell SageMaker where to find the script
        framework_version='1.3-1',
        py_version='py3',
        instance_type='ml.m5.large',
        instance_count=1,
        role=role,
        output_path=f's3://{project}/output',
        hyperparameters={
            'num_round': '10',
            'objective': 'binary:logistic'
        },
        sagemaker_session=sagemaker_session
    )
    
    # Start training
    estimator.fit({'train': f's3://{project}/train/'})
    print(f"Training job started: {estimator.latest_training_job.name}")
    return estimator.latest_training_job.name

if __name__ == '__main__':
    main()