import argparse
import boto3
from sagemaker import Session
from sagemaker.xgboost import XGBoostModel

def deploy_model(endpoint_name, model_data, role_arn, instance_type, region):
    session = Session(boto3.Session(region_name=region))
    
    model = XGBoostModel(
        model_data=model_data,
        role=role_arn,
        framework_version='1.5-1',
        sagemaker_session=session
    )
    
    try:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            wait=False
        )
        print(f"✅ Deployment initiated for endpoint: {endpoint_name}")
        return True
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', required=True)
    parser.add_argument('--model-data', required=True)
    parser.add_argument('--role-arn', required=True)
    parser.add_argument('--instance-type', default='ml.m5.large')
    parser.add_argument('--region', default='ap-south-1')
    
    args = parser.parse_args()
    
    if not deploy_model(
        args.endpoint_name,
        args.model_data,
        args.role_arn,
        args.instance_type,
        args.region
    ):
        exit(1)