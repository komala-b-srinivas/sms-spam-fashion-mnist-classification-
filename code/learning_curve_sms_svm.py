import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.svm import LinearSVC
from sms_spam import preprocess_spam_dataset
from scipy.sparse import vstack

def plot_learning_curve(estimator, title, X, y, cv, filename):
    os.makedirs("outputs", exist_ok=True)

    train_sizes = np.linspace(0.1, 1.0, 6)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='accuracy', train_sizes=train_sizes, n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-Validation Accuracy')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')

    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}.png")
    print(f" Saved plot: outputs/{filename}.png")
    plt.show()

def main():
    X_train, X_test, y_train, y_test, vectorizer = preprocess_spam_dataset()

    X = vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    model = LinearSVC(C=1.0, max_iter=1000)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    plot_learning_curve(
        estimator=model,
        title="Learning Curve - SVM (SMS Spam)",
        X=X,
        y=y,
        cv=cv,
        filename="learning_curve_sms_svm"
    )

if __name__ == "__main__":
    main()
