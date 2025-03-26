# Use official SageMaker base image or a lightweight Python image
FROM python:3.8-slim

# Set environment variables (SageMaker paths for training)
ENV PYTHONUNBUFFERED=TRUE \
    PATH="/opt/ml/code:/usr/local/bin:${PATH}" \
    TRAINING_PATH="/opt/ml/input/data/train.csv" \
    MODEL_PATH="/opt/ml/model"

# Install core dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create the SageMaker required folder structure
RUN mkdir -p /opt/ml/code /opt/ml/input/data /opt/ml/model
WORKDIR /opt/ml/code

# Copy training script and any additional dependencies
COPY train.py .

# Define the entry point for the container
ENTRYPOINT ["python3", "train.py"]
