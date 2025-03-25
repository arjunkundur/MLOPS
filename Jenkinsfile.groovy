pipeline {
    agent any

    environment {
        AWS_REGION = 'us-west-2'
        S3_BUCKET = 'jenkinsaws-mlops3030'
    }

    stages {
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python environment and installing required libraries...'
                sh '''
                # Use python3 and pip3 explicitly if pip is not found
                pip3 install --upgrade boto3 sagemaker pandas scikit-learn
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
                python3 sagemaker-pipelines-train-pipeline.py
                '''
            }
        }

        stage('Monitor SageMaker Pipeline') {
            steps {
                echo 'Monitoring the SageMaker pipeline status...'
                sh '''
                python3 monitor-pipeline-status.py
                '''
            }
        }

        stage('Perform Inference') {
            steps {
                echo 'Running inference on test data...'
                sh '''
                python3 sagemaker-inference.py
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
