import boto3
import argparse
import time
from sagemaker import image_uris

def get_latest_model(client, training_job_name):
    """Get model artifacts from training job"""
    training_job = client.describe_training_job(
        TrainingJobName=training_job_name
    )
    return training_job['ModelArtifacts']['S3ModelArtifacts']

def create_model(client, project_name, model_data, region, instance_type):
    """Create SageMaker model"""
    container_image = image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.2-1'
    )
    
    model_name = f"{project_name}-model-{int(time.time())}"
    
    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn="arn:aws:iam::975050337104:role/service-role/AmazonSageMaker-ExecutionRole-20250311T162664",
        Containers=[{
            'Image': container_image,
            'ModelDataUrl': model_data,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'train.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': model_data.replace('model.tar.gz', 'source/sourcedir.tar.gz')
            }
        }]
    )
    return model_name

def create_endpoint_config(client, project_name, model_name, instance_type):
    """Create endpoint configuration"""
    config_name = f"{project_name}-config-{int(time.time())}"
    
    client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': instance_type
        }]
    )
    return config_name

def deploy_endpoint(client, endpoint_name, config_name):
    """Deploy or update endpoint"""
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Updating existing endpoint: {endpoint_name}")
        client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except client.exceptions.ResourceNotFound:
        print(f"Creating new endpoint: {endpoint_name}")
        client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--training-job-name', required=True)
    parser.add_argument('--instance-type', default='ml.m5.large')
    args = parser.parse_args()
    
    client = boto3.client('sagemaker', region_name=args.region)
    endpoint_name = f"{args.project}-endpoint"
    
    # Get model artifacts
    model_data = get_latest_model(client, args.training_job_name)
    print(f"Model artifacts: {model_data}")
    
    # Create resources
    model_name = create_model(client, args.project, model_data, args.region, args.instance_type)
    config_name = create_endpoint_config(client, args.project, model_name, args.instance_type)
    
    # Deploy endpoint
    deploy_endpoint(client, endpoint_name, config_name)
    
    # Output for Jenkins parsing
    print("\n=== DEPLOYMENT SUMMARY ===")
    print(f"Endpoint: {endpoint_name}")
    print(f"EndpointConfig: {config_name}")
    print(f"Model: {model_name}")
    print(f"TrainingJob: {args.training_job_name}")
    print(f"InstanceType: {args.instance_type}")

if __name__ == "__main__":
    main()