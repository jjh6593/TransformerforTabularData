import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
def save_parameters(args):
    with open(os.path.join(args['save_folder'], 'parameters.txt'), 'w') as f:
        for key, value in args.items():
            f.write(f"{key}: {value}\n")
    print("Parameters saved to parameters.txt.")

def load_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Target'])
    y = df['Target']
    y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    float_cols = X.select_dtypes(include=['float', 'int']).columns
    num_numerical_cols = len(float_cols)

    scaler_x = MinMaxScaler()
    X[float_cols] = scaler_x.fit_transform(X[float_cols])
    X_numerical = torch.tensor(X[float_cols].values, dtype=torch.float32)

    return X_numerical, y, num_numerical_cols

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_device(use_cuda):
    return torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
