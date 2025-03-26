import boto3
import argparse
import time

def verify_training_job(client, training_job_name):
    """Verify the exact training job exists and is completed"""
    try:
        response = client.describe_training_job(
            TrainingJobName=training_job_name
        )
        if response['TrainingJobStatus'] != 'Completed':
            raise ValueError(f"Training job {training_job_name} status: {response['TrainingJobStatus']}")
        return response
    except client.exceptions.ResourceNotFound:
        raise ValueError(f"Training job {training_job_name} not found")

def deploy_from_specific_job(project_name, region, training_job_name):
    client = boto3.client('sagemaker', region_name=region)
    
    # 1. Verify the exact training job exists
    job_details = verify_training_job(client, training_job_name)
    model_data = job_details['ModelArtifacts']['S3ModelArtifacts']
    print(f"Using model from exact training job: {training_job_name}")
    print(f"Model artifacts: {model_data}")

    # 2. Create model
    model_name = f"{project_name}-model-{training_job_name[-8:]}"
    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn="arn:aws:iam::975050337104:role/service-role/AmazonSageMaker-ExecutionRole-20250311T162664",
        Containers=[{
            'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.2-1',
            'ModelDataUrl': model_data,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'train.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': model_data.replace('model.tar.gz', 'source/sourcedir.tar.gz')
            }
        }]
    )

    # 3. Create endpoint config
    config_name = f"{project_name}-config-{training_job_name[-8:]}"
    client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large'
        }]
    )

    # 4. Deploy endpoint
    endpoint_name = f"{project_name}-endpoint"
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Updating existing endpoint with model from {training_job_name}")
        client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except client.exceptions.ResourceNotFound:
        print(f"Creating new endpoint with model from {training_job_name}")
        client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    
    return endpoint_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--training-job-name', required=True)
    args = parser.parse_args()
    
    endpoint = deploy_from_specific_job(
        args.project,
        args.region,
        args.training_job_name
    )
    print(f"Deployed endpoint '{endpoint}' using model from {args.training_job_name}")

if __name__ == "__main__":
    main()