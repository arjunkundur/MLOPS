import argparse
import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='/opt/ml/input/data/train/train.csv', help='Path to training data')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model', help='Directory to save the trained model')
    return parser.parse_args()

def main():
    args = parse_args()

    print("Loading training data...")
    data = pd.read_csv(args.train_data)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.4f}")

    # Save the model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("Model saved.")

if __name__ == "__main__":
    main()
