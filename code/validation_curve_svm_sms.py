import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import validation_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os

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
        SVC(kernel="linear")
    )

    param_range = [0.001, 0.01, 0.1, 1, 10, 100]

    train_scores, val_scores = validation_curve(
        pipeline,
        X,
        y,
        param_name="svc__C",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_mean, label="Training Accuracy", marker='o')
    plt.plot(param_range, val_mean, label="Validation Accuracy", marker='o')
    plt.xscale('log')
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel("Accuracy")
    plt.title("Validation Curve – SVM (SMS Spam Dataset)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/validation_curve_svm_sms.png")
    print(" Saved: outputs/validation_curve_svm_sms.png")
    plt.show()

if __name__ == "__main__":
    main()
