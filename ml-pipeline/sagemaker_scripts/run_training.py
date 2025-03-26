import argparse
import boto3
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True, help='SageMaker project name')
    parser.add_argument('--region', type=str, required=True, help='AWS region')
    parser.add_argument('--role', type=str, required=True, help='SageMaker execution role ARN')
    args = parser.parse_args()

    # Initialize SageMaker client
    sm_client = boto3.client('sagemaker', region_name=args.region)

    # Define job name with timestamp to avoid collisions
    training_job_name = f"{args.project}-training-{int(time.time())}"

    # Define training job parameters with correctly prefixed hyperparameters
    training_params = {
        "TrainingJobName": training_job_name,
        "AlgorithmSpecification": {
            "TrainingImage": "975050337104.dkr.ecr.ap-south-1.amazonaws.com/xgboost:latest",
            "TrainingInputMode": "File",
        },
        "RoleArn": args.role,
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": f"s3://{args.project}/train",
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{args.project}/output",
        },
        "ResourceConfig": {
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},

        # Corrected hyperparameters with "--" prefix
        "HyperParameters": {
            "--train": "/opt/ml/input/data/train/train.csv",
            "--model-dir": "/opt/ml/model",
        }
    }

    # Start the training job on SageMaker
    response = sm_client.create_training_job(**training_params)
    
    # Print only the job name for Jenkins
    print(training_job_name)

if __name__ == '__main__':
    main()
