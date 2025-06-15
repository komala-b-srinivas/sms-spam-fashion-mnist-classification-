import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_fashion_mnist():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, '..', 'data', 'fashion-mnist_train.csv')
    test_path = os.path.join(base_path, '..', 'data', 'fashion-mnist_test.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
