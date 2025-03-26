import warnings
import sys
import logging
from sagemaker.xgboost import XGBoost

# Suppress all warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

def main():
    # Your training configuration - adjust these as needed
    estimator = XGBoost(
        entry_script="train.py",
        role="your-sagemaker-role-arn",
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.3-1"
    )
    
    # Start training job
    estimator.fit({"train": "s3://your-training-data-path"})
    
    # Print ONLY the job name with no extra output
    print(estimator.latest_training_job.name, end='')
    sys.stdout.flush()

if __name__ == "__main__":
    main()