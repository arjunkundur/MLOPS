import pandas as pd
import xgboost as xgb
import os
import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()
    
    # Load training data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    
    # Train model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=10,
        max_depth=5,
        learning_rate=0.2
    )
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(os.path.join(args.model_dir, 'xgboost-model.json'))

if __name__ == '__main__':
    train()