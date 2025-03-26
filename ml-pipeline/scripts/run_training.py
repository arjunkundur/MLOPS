import boto3
import time
import pprint

sm_client = boto3.client('sagemaker', region_name='ap-south-1')

project = "my-sagemaker-aws-jenkins-project1"
role = "arn:aws:iam::975050337104:role/service-role/AmazonSageMaker-ExecutionRole-20250311T162664"
training_job_name = f"{project}-training-{int(time.time())}"

training_params = {
    "TrainingJobName": training_job_name,
    "AlgorithmSpecification": {
        "TrainingImage": "975050337104.dkr.ecr.ap-south-1.amazonaws.com/xgboost:latest",
        "TrainingInputMode": "File",
    },
    "RoleArn": role,
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3Uri": f"s3://{project}/train",
                    "S3DataType": "S3Prefix",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "text/csv",
        }
    ],
    "OutputDataConfig": {
        "S3OutputPath": f"s3://{project}/output",
    },
    "ResourceConfig": {
        "InstanceType": "ml.m5.large",
        "InstanceCount": 1,
        "VolumeSizeInGB": 10,
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
    
    # Only include actual XGBoost hyperparameters here
    "HyperParameters": {
        "num_round": "10",
        "objective": "binary:logistic",
        "max_depth": "5",
        "eta": "0.2"
    }
}

# Print the parameters being passed to SageMaker
print("Sending the following parameters to SageMaker:")
pprint.pprint(training_params)

# Create the training job
response = sm_client.create_training_job(**training_params)
print(f"Training Job created: {training_job_name}")