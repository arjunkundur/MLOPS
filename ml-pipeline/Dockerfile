# Base image
FROM python:3.8-slim

# Install required system packages, including gcc and other build tools
RUN apt-get update && apt-get install -y gcc build-essential

# Install Python packages
RUN pip install --upgrade pip
RUN pip install xgboost pandas sagemaker-training

# Copy the training code
COPY train.py /opt/ml/code/train.py

# Set the working directory
WORKDIR /opt/ml/code

# Set the entry point for training
ENTRYPOINT ["python", "train.py"]
