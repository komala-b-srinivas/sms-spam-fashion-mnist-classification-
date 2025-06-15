import numpy as np
from sms_spam import preprocess_spam_dataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

def main():
    # Load and preprocess SMS Spam data
    X_train, X_test, y_train, y_test, tfidf = preprocess_spam_dataset()

    # Define Naive Bayes model and parameter grid
    model = MultinomialNB()
    param_grid = {
        'alpha': [0.01, 0.1, 1.0],
        'fit_prior': [True, False]
    }

    # Grid search with 5-fold cross-validation
    grid = GridSearchCV(model, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)

    # Best model
    best_model = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Final Accuracy:", round(acc, 4))
    print("Final F1 Score:", round(f1, 4))

if __name__ == "__main__":
    main()
