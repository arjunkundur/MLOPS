import argparse
import os
import pandas as pd
import xgboost as xgb
import json

def parse_args():
    # SageMaker passes hyperparameters as command-line arguments
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--num_round', type=int, default=10)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Loaded arguments:")
    print(json.dumps(vars(args), indent=2))
    
    # Load data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=args.num_round,
        objective=args.objective,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(os.path.join(args.model_dir, 'xgboost-model.json'))
    print("Model saved successfully")

if __name__ == '__main__':
    main()