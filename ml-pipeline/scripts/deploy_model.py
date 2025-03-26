import boto3
import argparse
import time
from sagemaker import image_uris  # Add this import at the top

def get_latest_model(client, training_job_name):
    """Get model artifacts from training job"""
    training_job = client.describe_training_job(
        TrainingJobName=training_job_name
    )
    return training_job['ModelArtifacts']['S3ModelArtifacts']

def create_endpoint_config(client, project_name, model_data, region):
    """Create endpoint configuration for the model"""
    config_name = f"{project_name}-config-{int(time.time())}"
    
    # Get the correct container image for the region
    container_image = image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.2-1'
    )
    
    client.create_model(
        ModelName=f"{project_name}-model",
        ExecutionRoleArn="arn:aws:iam::975050337104:role/service-role/AmazonSageMaker-ExecutionRole-20250311T162664",
        Containers=[{
            'Image': container_image,  # Use the retrieved image
            'ModelDataUrl': model_data,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'train.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': model_data.replace('model.tar.gz', 'source/sourcedir.tar.gz')
            }
        }]
    )
    
    client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': f"{project_name}-model",
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.m5.large'
        }]
    )
    return config_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='Project name')
    parser.add_argument('--region', required=True, help='AWS Region')
    parser.add_argument('--training-job-name', required=True, help='Training job name')
    args = parser.parse_args()
    
    client = boto3.client('sagemaker', region_name=args.region)
    
    # Get model artifacts from training job
    model_data = get_latest_model(client, args.training_job_name)
    print(f"Found model artifacts: {model_data}")
    
    # Create endpoint configuration
    config_name = create_endpoint_config(client, args.project, model_data, args.region)
    print(f"Created endpoint config: {config_name}")
    
    # Create or update endpoint
    endpoint_name = f"{args.project}-endpoint"
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Updating existing endpoint {endpoint_name}")
        client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except client.exceptions.ResourceNotFound:
        print(f"Creating new endpoint {endpoint_name}")
        client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    
    print(f"Deployment initiated for model from job {args.training_job_name}")

if __name__ == "__main__":
    main()