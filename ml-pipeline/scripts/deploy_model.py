import boto3
import sys
import time
from botocore.exceptions import ClientError

def endpoint_exists(client, endpoint_name):
    """Safely check if endpoint exists"""
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as e:
        if "Could not find endpoint" in str(e):
            return False
        raise

def wait_for_endpoint(client, endpoint_name, max_attempts=30, delay=30):
    """Wait for endpoint to be InService"""
    print(f"⌛ Waiting for endpoint {endpoint_name} to be InService...")
    waiter = client.get_waiter('endpoint_in_service')
    try:
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={
                'MaxAttempts': max_attempts,
                'Delay': delay
            }
        )
        print(f"✅ Endpoint {endpoint_name} is now InService")
        return True
    except ClientError as e:
        print(f"❌ Error while waiting for endpoint: {str(e)}")
        return False

def deploy_endpoint(client, endpoint_name, config_name, tags=None):
    """
    Create or update endpoint with proper existence check
    Args:
        client: boto3 SageMaker client
        endpoint_name: Name of the endpoint
        config_name: Endpoint config name to use
        tags: Optional list of tags to apply
    Returns:
        dict: AWS response
    """
    try:
        if endpoint_exists(client, endpoint_name):
            print(f"🔄 Updating existing endpoint: {endpoint_name}")
            response = client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        else:
            print(f"🆕 Creating new endpoint: {endpoint_name}")
            create_args = {
                'EndpointName': endpoint_name,
                'EndpointConfigName': config_name
            }
            if tags:
                create_args['Tags'] = tags
            response = client.create_endpoint(**create_args)
        
        # Wait for deployment to complete
        if not wait_for_endpoint(client, endpoint_name):
            raise RuntimeError(f"Endpoint {endpoint_name} failed to reach InService status")
        
        return response
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceLimitExceeded':
            print("❌ Account limit reached for endpoints")
        elif error_code == 'ResourceInUse':
            print("❌ Endpoint is currently updating")
        raise

def validate_config_exists(client, config_name):
    """Verify endpoint config exists before deployment"""
    try:
        client.describe_endpoint_config(EndpointConfigName=config_name)
        return True
    except ClientError as e:
        print(f"❌ Endpoint config {config_name} does not exist")
        return False

def main():
    # Initialize boto3 client
    client = boto3.client('sagemaker')
    
    # Example configuration - replace with your actual args parsing
    endpoint_name = "my-sagemaker-aws-jenkins-project1-endpoint"
    config_name = "your-endpoint-config-name"
    tags = [
        {'Key': 'Project', 'Value': 'jenkins-project1'},
        {'Key': 'Environment', 'Value': 'production'}
    ]
    
    # Validate config exists first
    if not validate_config_exists(client, config_name):
        sys.exit(1)
    
    # Deploy endpoint
    try:
        print(f"🚀 Starting endpoint deployment for {endpoint_name}")
        response = deploy_endpoint(client, endpoint_name, config_name, tags)
        
        # Get final endpoint status
        endpoint_info = client.describe_endpoint(EndpointName=endpoint_name)
        print(f"\n🎉 Deployment successful!")
        print(f"Endpoint ARN: {response['EndpointArn']}")
        print(f"Endpoint Status: {endpoint_info['EndpointStatus']}")
        print(f"Endpoint URL: https://runtime.sagemaker.{boto3.session.Session().region_name}.amazonaws.com/endpoints/{endpoint_name}/invocations")
        
    except Exception as e:
        print(f"\n💥 Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()