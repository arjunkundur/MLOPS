import boto3
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

def deploy_endpoint(client, endpoint_name, config_name):
    """Create or update endpoint with proper existence check"""
    if endpoint_exists(client, endpoint_name):
        print(f"Updating existing endpoint: {endpoint_name}")
        return client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    else:
        print(f"Creating new endpoint: {endpoint_name}")
        return client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )

def main():
    # ... [previous code remains the same until endpoint deployment]
    
    # Deploy endpoint
    endpoint_name = f"{args.project}-endpoint"
    try:
        response = deploy_endpoint(client, endpoint_name, config_name)
        print(f"✅ Endpoint deployment initiated: {endpoint_name}")
        print(f"Endpoint ARN: {response['EndpointArn']}")
    except Exception as e:
        print(f"❌ Endpoint deployment failed: {str(e)}")
        raise