import numpy as np
from fashion_mnist import preprocess_fashion_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

def main():
    # Load data
    X_train, X_test, y_train, y_test = preprocess_fashion_mnist()

    # Use only 10,000 training samples for faster convergence
    X_train = X_train[:10000]
    y_train = y_train[:10000]

    # Scale data using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression with solver 'lbfgs'
    model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0,
        max_iter=500
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Final Accuracy:", round(acc, 4))
    print("Final Macro F1 Score:", round(f1, 4))

if __name__ == "__main__":
    main()



