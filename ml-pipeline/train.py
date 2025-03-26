import pandas as pd
import os
import xgboost as xgb

def model_training():
    # SageMaker default paths
    input_path = '/opt/ml/input/data/train/'
    model_path = '/opt/ml/model/'
    
    # Load training data (assuming file is named train.csv)
    data = pd.read_csv(os.path.join(input_path, 'train.csv'))
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Train model
    model = xgb.XGBClassifier()
    model.fit(X, y)

    # Save model
    model.save_model(os.path.join(model_path, "xgboost-model.json"))

if __name__ == '__main__':
    model_training()