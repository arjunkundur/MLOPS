pipeline {
    agent {
        // Use a Docker agent with Python pre-installed
        docker {
            image 'python:3.9-slim'
        }
    }
    environment {
        AWS_REGION = 'us-west-2'  // Set your AWS region
        S3_BUCKET = 'jenkinsaws-mlops3030'
    }
    stages {
        stage('Setup Environment') {
            steps {
                echo 'Installing required Python libraries...'
                sh '''
                pip install --upgrade boto3 sagemaker pandas scikit-learn
                '''
            }
        }
        stage('Clone Repository') {
            steps {
                echo 'Cloning Git repository with SageMaker scripts...'
                git branch: 'main', url: 'https://github.com/arjunkundur/MLOPS.git'
            }
        }
        stage('Run SageMaker Pipeline') {
            steps {
                echo 'Executing SageMaker training pipeline...'
                sh '''
                python sagemaker-pipelines-train-pipeline.py
                '''
            }
        }
        stage('Monitor SageMaker Pipeline') {
            steps {
                echo 'Monitoring the SageMaker pipeline status...'
                sh '''
                python monitor-pipeline-status.py
                '''
            }
        }
        stage('Perform Inference') {
            steps {
                echo 'Running inference on test data...'
                sh '''
                python sagemaker-inference.py
                '''
            }
        }
    }
    post {
        success {
            echo 'SageMaker pipeline completed successfully!'
        }
        failure {
            echo 'SageMaker pipeline failed. Check logs for more details.'
        }
    }
}
