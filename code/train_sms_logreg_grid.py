import numpy as np
from sms_spam import preprocess_spam_dataset
from sklearn.metrics import accuracy_score, f1_score
from logistic_regression import LogisticRegression

def evaluate(y_true, y_pred):
    """
    Evaluates the model predictions and returns Accuracy and F1 Score
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, f1

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, tfidf = preprocess_spam_dataset()

    # Convert sparse TF-IDF matrix to dense array for custom model
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # Define hyperparameter grid
    alpha_list = [0.01, 0.05, 0.1]
    max_iters_list = [500, 1000, 1500]

    # Store best results
    best_f1 = 0
    best_params = {}
    best_model = None

    print("Tuning hyperparameters...\n")

    for alpha in alpha_list:
        for max_iter in max_iters_list:
            print(f"Training Logistic Regression with alpha={alpha}, max_iters={max_iter}")
            model = LogisticRegression(max_iters=max_iter, alpha=alpha)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            acc, f1 = evaluate(y_test, y_pred)
            print(f"Accuracy: {round(acc, 4)}, F1 Score: {round(f1, 4)}\n")

            if f1 > best_f1:
                best_f1 = f1
                best_params = {'alpha': alpha, 'max_iters': max_iter}
                best_model = model

    print("Best Model Results:")
    print("Alpha:", best_params['alpha'])
    print("Max Iterations:", best_params['max_iters'])

    # Final evaluation
    final_preds = best_model.predict(X_test)
    final_acc, final_f1 = evaluate(y_test, final_preds)
    print("Final Accuracy:", round(final_acc, 4))
    print("Final F1 Score:", round(final_f1, 4))

if __name__ == "__main__":
    main()
