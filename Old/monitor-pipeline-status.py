import boto3
import time

# Define constants
pipeline_name = "ark-mlops-jenkins"  # Update with your SageMaker pipeline name
region = "us-west-2"  # Your AWS region
check_interval = 60  # Time in seconds between status checks

# Initialize SageMaker client
sagemaker_client = boto3.client("sagemaker", region_name=region)

def get_pipeline_status(pipeline_name):
    """Fetch the current status of the specified SageMaker pipeline."""
    response = sagemaker_client.list_pipeline_executions(PipelineName=pipeline_name)
    if "PipelineExecutionSummaries" in response and len(response["PipelineExecutionSummaries"]) > 0:
        return response["PipelineExecutionSummaries"][0]["PipelineExecutionStatus"]
    else:
        return None

def monitor_pipeline(pipeline_name):
    """Monitor the SageMaker pipeline until it finishes."""
    print(f"Monitoring pipeline '{pipeline_name}' in region '{region}'...")

    while True:
        status = get_pipeline_status(pipeline_name)
        if status:
            print(f"Pipeline Execution Status: {status}")
            if status in ["Succeeded", "Failed", "Stopped"]:
                break
        else:
            print("No active pipeline executions found.")
            break

        time.sleep(check_interval)

if __name__ == "__main__":
    monitor_pipeline(pipeline_name)
