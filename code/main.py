# ===============================
# CSC 272 Course Project - main.py
# Name: Komala Belur Srinivas and Ananya Purohith
# Description:
# This script evaluates final tuned models for both SMS Spam and Fashion MNIST datasets.
# It uses sklearn implementations for consistency and plotting outputs.
# Note: For SMS Spam, custom implementations of Logistic Regression and Perceptron
# were used during training (in separate scripts) and results reused here where applicable.
# ===============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sms_spam import preprocess_spam_dataset
from fashion_mnist import preprocess_fashion_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import os
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Function to generate and save confusion matrix as a plot
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}")
    plt.close()

# Helper function to compute accuracy, F1, and save confusion matrix
def evaluate_and_save(name, model, X_test, y_test, task='binary'):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) if task == 'binary' else f1_score(y_test, y_pred, average='macro')
    print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    save_confusion_matrix(y_test, y_pred, title=name, filename=f"conf_matrix_{name.replace(' ', '_')}.png")
    return name, acc, f1

# ==================== SMS SPAM ====================
print("\n=== SMS Spam Dataset ===")
X_train_s, X_test_s, y_train_s, y_test_s, tfidf = preprocess_spam_dataset()

results_spam = []

# Logistic Regression (sklearn version for consistency in this script)
logreg = LogisticRegression(C=1.0, solver='liblinear', max_iter=200)
logreg.fit(X_train_s, y_train_s)
results_spam.append(evaluate_and_save("Logistic Regression", logreg, X_test_s, y_test_s))

# KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_s.toarray(), y_train_s)
results_spam.append(evaluate_and_save("KNN", knn, X_test_s.toarray(), y_test_s))

# Naive Bayes
nb = MultinomialNB(alpha=0.1, fit_prior=True)
nb.fit(X_train_s, y_train_s)
results_spam.append(evaluate_and_save("Naive Bayes", nb, X_test_s, y_test_s))

# SVM
svm = SVC(C=1, kernel='linear')
svm.fit(X_train_s, y_train_s)
results_spam.append(evaluate_and_save("SVM", svm, X_test_s, y_test_s))

# ==================== FASHION MNIST ====================
print("\n=== Fashion MNIST Dataset ===")
X_train_f, X_test_f, y_train_f, y_test_f = preprocess_fashion_mnist()

results_fashion = []

# Logistic Regression
logreg_f = LogisticRegression(C=1.0, solver='saga', multi_class='multinomial', max_iter=200)
logreg_f.fit(X_train_f, y_train_f)
results_fashion.append(evaluate_and_save("Logistic Regression (Fashion)", logreg_f, X_test_f, y_test_f, task='multi'))

# KNN
knn_f = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_f.fit(X_train_f, y_train_f)
results_fashion.append(evaluate_and_save("KNN (Fashion)", knn_f, X_test_f, y_test_f, task='multi'))

# Decision Tree
tree_f = DecisionTreeClassifier(max_depth=20, min_samples_split=2)
tree_f.fit(X_train_f, y_train_f)
results_fashion.append(evaluate_and_save("Decision Tree (Fashion)", tree_f, X_test_f, y_test_f, task='multi'))

# MLP
mlp_f = MLPClassifier(hidden_layer_sizes=(128,), learning_rate_init=0.001, max_iter=200)
mlp_f.fit(X_train_f, y_train_f)
results_fashion.append(evaluate_and_save("MLP (Fashion)", mlp_f, X_test_f, y_test_f, task='multi'))

# ==================== BAR PLOTS ====================
# Generate bar plots for model performance

def plot_bar(results, title, filename):
    models, accs, f1s = zip(*results)
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, accs, width, label='Accuracy')
    plt.bar(x + width/2, f1s, width, label='F1 Score')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}")
    plt.close()

plot_bar(results_spam, "Model Performance on SMS Spam", "sms_performance.png")
plot_bar(results_fashion, "Model Performance on Fashion MNIST", "fashion_performance.png")
