import numpy as np
from fashion_mnist import preprocess_fashion_mnist
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

def main():
    """
    Trains and evaluates a Multi-layer Perceptron on Fashion MNIST using GridSearchCV.
    """

    # Load and preprocess dataset
    X_train, X_test, y_train, y_test = preprocess_fashion_mnist()

    # Define model and parameter grid
    model = MLPClassifier(max_iter=200)  # Keep small for faster tuning
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,)],
        'learning_rate_init': [0.001, 0.01]
    }

    # Perform grid search
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=3)
    grid.fit(X_train, y_train)

    # Best model and parameters
    best_model = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Final Accuracy:", round(acc, 4))
    print("Final Macro F1 Score:", round(f1, 4))

if __name__ == "__main__":
    main()
