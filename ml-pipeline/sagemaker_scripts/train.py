import argparse
import pandas as pd
import os
import xgboost as xgb

def model_training(train_data_path, model_output_path):
    # Load and preprocess data
    data = pd.read_csv(train_data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Train a basic XGBoost model
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # Save the trained model
    model.save_model(os.path.join(model_output_path, "xgboost-model.json"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory to save the trained model')
    args = parser.parse_args()

    # Start model training
    model_training(train_data_path=args.train, model_output_path=args.model_dir)
