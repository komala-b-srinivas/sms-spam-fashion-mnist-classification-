import numpy as np
from sms_spam import preprocess_spam_dataset
from sklearn.metrics import accuracy_score, f1_score
from perceptron import Perceptron  # Make sure this file exists!

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

    # Convert sparse matrix to dense
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # Perceptron expects labels as -1 and 1 (not 0 and 1)
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)

    # Try different max_iters to simulate hyperparameter tuning
    max_iters_list = [100, 500, 1000]
    best_f1 = 0
    best_model = None
    best_params = {}

    print("Tuning Perceptron hyperparameters...\n")

    for max_iter in max_iters_list:
        print(f"Training Perceptron with max_iters={max_iter}")
        model = Perceptron(max_iters=max_iter)
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        acc, f1 = evaluate(y_test, y_pred)
        print(f"Accuracy: {round(acc, 4)}, F1 Score: {round(f1, 4)}\n")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = {'max_iters': max_iter}

    print("Best Perceptron Model:")
    print("Max Iterations:", best_params['max_iters'])

    final_preds = best_model.predict(X_test)
    final_acc, final_f1 = evaluate(y_test, final_preds)
    print("Final Accuracy:", round(final_acc, 4))
    print("Final F1 Score:", round(final_f1, 4))

if __name__ == "__main__":
    main()
