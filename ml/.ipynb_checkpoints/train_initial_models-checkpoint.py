# =========================================
# Final ML Training Script - Corrected
# =========================================

import pandas as pd
from spark_session import spark
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# -------------------------------
# 1. Load dataset (FULL DATA)
# -------------------------------
df_spark = spark.read.parquet("hdfs://localhost:9000/processed/final_dataset")
df = df_spark.toPandas()

# -------------------------------
# 2. Handle missing clean_text
# -------------------------------
df['clean_text'] = df['clean_text'].fillna("")  # replace None/NaN with empty string
print("Missing 'clean_text' after fillna:", df['clean_text'].isna().sum())

# -------------------------------
# 3. Generate sentiment labels
# -------------------------------
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return 1       # Positive
    elif polarity < -0.1:
        return -1      # Negative
    else:
        return 0       # Neutral

df['sentiment'] = df['clean_text'].apply(get_sentiment)
print("Original distribution:\n", df['sentiment'].value_counts())

# -------------------------------
# 4. Balance dataset
# -------------------------------
df_neg = df[df.sentiment == -1]
df_neu = df[df.sentiment == 0].sample(len(df_neg), random_state=42)
df_pos = df[df.sentiment == 1]

df_balanced = pd.concat([df_neg, df_neu, df_pos]).sample(frac=1, random_state=42)
print("Balanced distribution:\n", df_balanced['sentiment'].value_counts())

# -------------------------------
# 5. Train TF-IDF Vectorizer
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words=None,   # <- allow all words including 'o'
    min_df=1,          # <- include rare words
    sublinear_tf=True
)


X = vectorizer.fit_transform(df_balanced["clean_text"])
y = df_balanced["sentiment"]

# -------------------------------
# 6. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# 7. Train models
# -------------------------------
lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)
nb = MultinomialNB()
svm = LinearSVC(class_weight="balanced", C=1.5, max_iter=5000)

print("Training Logistic Regression...")
lr.fit(X_train, y_train)

print("Training Naive Bayes...")
nb.fit(X_train, y_train)

print("Training Linear SVM...")
svm.fit(X_train, y_train)

# -------------------------------
# 8. Evaluate
# -------------------------------
print("\nModel evaluation on test set:")
print("LR Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("NB Accuracy:", accuracy_score(y_test, nb.predict(X_test)))
print("SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test)))

# -------------------------------
# 9. Save models & vectorizer
# -------------------------------
os.makedirs("submit_dashboard/models", exist_ok=True)

joblib.dump(vectorizer, "submit_dashboard/models/tfidf_vectorizer.pkl")
joblib.dump(lr, "submit_dashboard/models/model_lr_final.pkl")
joblib.dump(nb, "submit_dashboard/models/model_nb_final.pkl")
joblib.dump(svm, "submit_dashboard/models/model_svm_final.pkl")

print("\n Models and vectorizer saved successfully!")
