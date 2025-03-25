import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model when the container is started
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

# Perform inference
def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions
