import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from fashion_mnist import preprocess_fashion_mnist
import warnings
from sklearn.exceptions import ConvergenceWarning

# Optional: suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    os.makedirs("outputs", exist_ok=True)

    # Load and prepare data
    X_train, X_test, y_train, y_test = preprocess_fashion_mnist()
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define grid: try different hidden layer configurations
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64), (128, 128)]
    }

    mlp = MLPClassifier(
        max_iter=100,
        solver='adam',
        random_state=42
    )

    grid = GridSearchCV(
        mlp,
        param_grid,
        cv=3,
        scoring='accuracy',
        return_train_score=True,
        n_jobs=-1
    )
    print("⏳ Running GridSearchCV...")
    grid.fit(X_scaled, y)
    print(" Grid search completed.")

    # Plotting results
    mean_scores = grid.cv_results_['mean_test_score']
    std_scores = grid.cv_results_['std_test_score']
    labels = ['-'.join(map(str, p['hidden_layer_sizes'])) for p in grid.cv_results_['params']]

    plt.figure(figsize=(8, 5))
    plt.errorbar(labels, mean_scores, yerr=std_scores, fmt='o-', capsize=5, color='darkorange')
    plt.title("MLP Hidden Layer Tuning (Fashion-MNIST)")
    plt.xlabel("Hidden Layer Sizes")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/tuning_curve_mlp_fashion.png")
    print(" Saved tuning curve: outputs/tuning_curve_mlp_fashion.png")
    plt.show()

if __name__ == "__main__":
    main()

