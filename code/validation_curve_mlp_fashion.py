import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def preprocess_fashion_mnist():
    # Dynamically build paths to CSV files
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, '..', 'data', 'fashion-mnist_train.csv')
    test_path = os.path.join(base_path, '..', 'data', 'fashion-mnist_test.csv')

    # Read datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features and labels
    X_train = train_df.drop("label", axis=1).values
    y_train = train_df["label"].values
    X_test = test_df.drop("label", axis=1).values
    y_test = test_df["label"].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    os.makedirs("outputs", exist_ok=True)

    # Load and combine data
    X_train, X_test, y_train, y_test = preprocess_fashion_mnist()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # Define alpha values for regularization
    param_range = [0.0001, 0.001, 0.01, 0.1, 1]

    # Run validation curve
    train_scores, val_scores = validation_curve(
        MLPClassifier(hidden_layer_sizes=(128,), solver='adam', max_iter=500, random_state=42),
        X, y,
        param_name="alpha",
        param_range=param_range,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    # Compute mean scores
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_mean, label="Training Accuracy", marker='o')
    plt.plot(param_range, val_mean, label="Validation Accuracy", marker='o')
    plt.xscale('log')
    plt.xlabel("Alpha (Regularization Strength)")
    plt.ylabel("Accuracy")
    plt.title("Validation Curve – MLP (Fashion-MNIST)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join("outputs", "validation_curve_mlp_fashion.png")
    plt.savefig(save_path)
    print(f" Saved: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
