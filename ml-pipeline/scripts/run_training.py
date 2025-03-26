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

    # Define training job parameters with updated HyperParameters
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
                "ContentType": "text/csv",  # Define input content type
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

        # Adding correct hyperparameters for train.py
        "HyperParameters": {
            "--train": "/opt/ml/input/data/train/data.csv",   # This correctly maps to `--train` argument in train.py
            "--model-dir": "/opt/ml/model",                   # This correctly maps to `--model-dir` argument
        }
    }

    # Start the training job
    response = sm_client.create_training_job(**training_params)
    
    # Print training job name
    print(f"Training job initiated: {training_job_name}")

if __name__ == '__main__':
    main()
