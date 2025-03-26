import pandas as pd
import os
import xgboost as xgb
import argparse

def model_training(train_data_path, model_output_path):
    try:
        # Explicitly check for sklearn
        from sklearn.base import is_classifier
        print("scikit-learn is available")
    except ImportError:
        print("ERROR: scikit-learn is required but not installed")
        raise

    try:
        # Load and preprocess data
        data = pd.read_csv(train_data_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Initialize XGBoost classifier with updated parameters
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=10,
            max_depth=5,
            learning_rate=0.2
        )
        
        # Train model
        model.fit(X, y)

        # Ensure output directory exists
        os.makedirs(model_output_path, exist_ok=True)
        
        # Save model
        model.save_model(os.path.join(model_output_path, "xgboost-model.json"))
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    args = parser.parse_args()
    model_training(args.train, args.model_dir)