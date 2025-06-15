import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])

def preprocess_spam_dataset():
    # Dynamically resolve the full path to data/spam.csv
    base_path = os.path.dirname(os.path.abspath(__file__))  # folder: /code
    data_path = os.path.join(base_path, '..', 'data', 'spam.csv')

    df = pd.read_csv(data_path, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['message'] = df['message'].apply(clean_text)

    le = LabelEncoder()
    y = le.fit_transform(df['label'])  # 0 = ham, 1 = spam

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['message'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, tfidf

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, tfidf = preprocess_spam_dataset()
    print("Preprocessing done. Number of training samples:", X_train.shape[0])
