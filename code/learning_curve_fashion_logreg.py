import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from fashion_mnist import preprocess_fashion_mnist
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def plot_learning_curve(estimator, title, X, y, cv, filename):
    os.makedirs("outputs", exist_ok=True)
    print("⏳ Generating learning curve...")

    train_sizes = np.linspace(0.1, 1.0, 6)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='accuracy', train_sizes=train_sizes, n_jobs=-1
    )

    print("✅ Learning curve computed.")
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Accuracy')
    plt.plot(train_sizes, test_mean, 'o-', label='Validation Accuracy')
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}.png")
    print(f"Saved plot: outputs/{filename}.png")
    plt.show()

def main():
    X_train, X_test, y_train, y_test = preprocess_fashion_mnist()

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=2000,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    plot_learning_curve(
        estimator=model,
        title="Learning Curve - Logistic Regression (Fashion MNIST)",
        X=X_scaled,
        y=y,
        cv=cv,
        filename="learning_curve_fashion_logreg"
    )

if __name__ == "__main__":
    main()
