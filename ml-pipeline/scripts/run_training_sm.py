import boto3
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.session import Session
import time

def main():
    # Configuration
    project = "my-sagemaker-aws-jenkins-project1"
    role = "arn:aws:iam::975050337104:role/service-role/AmazonSageMaker-ExecutionRole-20250311T162664"
    region = 'ap-south-1'
    
    # Initialize session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(boto_session=boto_session)
    
    # Create estimator
    estimator = XGBoost(
        entry_script='train.py',
        framework_version='1.3-1',  # Latest stable version
        py_version='py3',
        instance_type='ml.m5.large',
        instance_count=1,
        role=role,
        output_path=f's3://{project}/output',
        base_job_name=f'{project}-training',
        hyperparameters={
            'num_round': '10',
            'objective': 'binary:logistic'
        },
        sagemaker_session=sagemaker_session
    )
    
    # Start training
    training_job_name = f'{project}-training-{int(time.time())}'
    estimator.fit(
        {'train': f's3://{project}/train/'},
        job_name=training_job_name
    )
    
    print(f"Training job started: {training_job_name}")
    return training_job_name

if __name__ == '__main__':
    main()