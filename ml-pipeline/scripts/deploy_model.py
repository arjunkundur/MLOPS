import boto3
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--training-job-name', required=True)
    parser.add_argument('--instance-type', default='ml.m5.large')
    args = parser.parse_args()
    
    # Implementation from previous example
    # ...