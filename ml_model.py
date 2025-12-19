# ml_model.py
import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Ensure NLTK data is available
nltk.download('movie_reviews')

# 1. Load data
documents = []
labels = []
for fileid in movie_reviews.fileids():
    words = movie_reviews.raw(fileid)
    documents.append(words)
    label = movie_reviews.categories(fileid)[0]  # 'pos' or 'neg'
    labels.append(1 if label == 'pos' else 0)

# 2. Shuffle and split
combined = list(zip(documents, labels))
random.shuffle(combined)
documents, labels = zip(*combined)
documents = list(documents)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42, stratify=labels)

# 3. Vectorize (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"Validation Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Save model + vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sentiment_model.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
print("Saved model to models/sentiment_model.joblib and vectorizer to models/tfidf_vectorizer.joblib")
