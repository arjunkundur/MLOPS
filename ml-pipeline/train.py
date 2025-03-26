import os
import pandas as pd
import xgboost as xgb
import argparse

def model_training(train_data_path, model_output_path):
    # Load the dataset
    print("Loading data...")
    data = pd.read_csv(train_data_path)
    
    # Prepare features and labels
    X = data.drop(columns=["churn"])
    y = data["churn"]
    
    # Convert to DMatrix format (required by XGBoost)
    dtrain = xgb.DMatrix(X, label=y)
    
    # Define XGBoost parameters
    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.1,
        "eval_metric": "logloss",
    }
    
    # Train the XGBoost model
    print("Training the XGBoost model...")
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Save the model
    print(f"Saving the model to {model_output_path}")
    model.save_model(os.path.join(model_output_path, "xgboost-model.bin"))
    print("Model saved!")

if __name__ == "__main__":
    # Parse command-line arguments for SageMaker paths
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train", help="Training data path")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model", help="Path to save the trained model")
    
    args = parser.parse_args()
    
    # Training and saving the model
    model_training(train_data_path=os.path.join(args.train, "data.csv"), model_output_path=args.model_dir)
