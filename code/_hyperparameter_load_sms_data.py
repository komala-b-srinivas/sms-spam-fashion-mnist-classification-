import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline


def load_sms_data():
    import os
    # Dynamically get path to data/spam.csv
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data', 'spam.csv')

    df = pd.read_csv(data_path, encoding='latin-1')[["v1", "v2"]]
    df.columns = ["label", "text"]
    X = df["text"]
    y = LabelEncoder().fit_transform(df["label"])  # ham -> 0, spam -> 1
    return X, y


def main():
    os.makedirs("outputs", exist_ok=True)

    X, y = load_sms_data()

    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english'),
        SVC(kernel='linear')
    )

    param_grid = {'svc__C': [0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)

    scores = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    C_vals = [str(p['svc__C']) for p in grid.cv_results_['params']]

    plt.figure(figsize=(8, 5))
    plt.errorbar(C_vals, scores, yerr=stds, fmt='o-', capsize=5, color='purple')
    plt.title("SVM Hyperparameter Tuning Curve (SMS Spam Dataset)")
    plt.xlabel("Regularization Parameter C")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/tuning_curve_svm_sms.png")
    print(" Saved tuning curve: outputs/tuning_curve_svm_sms.png")
    plt.show()


if __name__ == "__main__":
    main()
