import boto3
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='Project name')
    parser.add_argument('--role', required=True, help='IAM Role ARN')
    parser.add_argument('--region', required=True, help='AWS Region')
    return parser.parse_args()

def main():
    args = parse_args()
    client = boto3.client('sagemaker', region_name=args.region)

    response = client.create_training_job(
        TrainingJobName=f"{args.project}-training",
        AlgorithmSpecification={
            'TrainingImage': '763104351884.dkr.ecr.ap-south-1.amazonaws.com/xgboost:latest',
            'TrainingInputMode': 'File'
        },
        RoleArn=args.role,
        OutputDataConfig={'S3OutputPath': f's3://{args.project}-output/'},
        ResourceConfig={
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 10
        },
        InputDataConfig=[
            {
                'ChannelName': 'train',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{args.project}-train-data/',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv'
            }
        ],
        StoppingCondition={'MaxRuntimeInSeconds': 3600}
    )
    print("Training job initiated:", response)

if __name__ == "__main__":
    main()
