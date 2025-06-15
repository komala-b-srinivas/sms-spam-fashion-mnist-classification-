import numpy as np
from sms_spam import preprocess_spam_dataset
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

def main():
    """
    Trains and evaluates SVM on the SMS Spam dataset using GridSearchCV.
    """

    # Load and preprocess dataset
    X_train, X_test, y_train, y_test, tfidf = preprocess_spam_dataset()

    # Define model and parameter grid
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear']  # linear kernel is best for text data
    }

    # Perform grid search with 5-fold CV using F1 Score
    grid = GridSearchCV(model, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    # Best model and parameters
    best_model = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)

    # Evaluate best model on test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Final Accuracy:", round(acc, 4))
    print("Final F1 Score:", round(f1, 4))

if __name__ == "__main__":
    main()
