import boto3
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='Project name')
    parser.add_argument('--region', required=True, help='AWS Region')
    return parser.parse_args()

def main():
    args = parse_args()
    client = boto3.client('sagemaker', region_name=args.region)

    endpoint_name = f"{args.project}-endpoint"
    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=f"{args.project}-endpoint-config"
    )
    print(f"Model deployed at endpoint: {endpoint_name}")

if __name__ == "__main__":
    main()
