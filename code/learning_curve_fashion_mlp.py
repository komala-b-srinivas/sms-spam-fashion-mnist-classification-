import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from fashion_mnist import preprocess_fashion_mnist

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def plot_learning_curve(estimator, title, X, y, cv, filename):
    import time
    os.makedirs("outputs", exist_ok=True)
    start = time.time()

    train_sizes = np.linspace(0.1, 1.0, 6)
    print("⏳ Generating learning curve...")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='accuracy', train_sizes=train_sizes, n_jobs=-1
    )

    print(f" Learning curve data ready in {round(time.time() - start, 2)} seconds.")
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

    path = f"outputs/{filename}.png"
    plt.savefig(path)
    print(f" Saved plot: {path}")
    plt.show()


def main():
    X_train, X_test, y_train, y_test = preprocess_fashion_mnist()

    # Combine and scale
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=500,  # ← increase from 100 to 300 or 500
        solver='adam',
        learning_rate_init=0.001,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    plot_learning_curve(
        estimator=model,
        title="Learning Curve - MLP (Fashion MNIST)",
        X=X_scaled,
        y=y,
        cv=cv,
        filename="learning_curve_fashion_mlp"
    )

if __name__ == "__main__":
    main()
