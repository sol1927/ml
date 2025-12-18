import pandas as pd
from spark_session import spark
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from textblob import TextBlob

# -------------------------------
# Load final dataset
# -------------------------------
df_spark = spark.read.parquet("hdfs://localhost:9000/processed/final_dataset")
df = df_spark.limit(1000).toPandas()  # sample for faster training

# -------------------------------
# Ensure 'clean_text' exists
# -------------------------------
if 'clean_text' not in df.columns:
    raise ValueError("Column 'clean_text' not found in dataset")

# -------------------------------
# Compute sentiment if missing
# -------------------------------
if 'sentiment' not in df.columns or df['sentiment'].isnull().all():
    def get_sentiment(text):
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0:
            return 1
        elif polarity < 0:
            return -1
        else:
            return 0

    df['sentiment'] = df['clean_text'].apply(get_sentiment)

# Class distribution
print("Sentiment distribution:\n", df['sentiment'].value_counts())

# -------------------------------
# Load TF-IDF vectorizer
# -------------------------------
vectorizer = joblib.load("submit_dashboard/models/tfidf_vectorizer.pkl")
X = vectorizer.transform(df["clean_text"])
y = df["sentiment"]

# -------------------------------
# Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train models
# -------------------------------

# Logistic Regression (multiclass compatible)
lr = LogisticRegression(
    max_iter=200,
    solver="lbfgs",      # works for multiclass
    class_weight="balanced"
)
lr.fit(X_train, y_train)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Linear SVM
svm = LinearSVC(class_weight="balanced", max_iter=2000)
svm.fit(X_train, y_train)

# -------------------------------
# Evaluate models
# -------------------------------
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb.predict(X_test)))
print("Linear SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test)))

# -------------------------------
# Save models
# -------------------------------
os.makedirs("submit_dashboard/models", exist_ok=True)
joblib.dump(lr, "submit_dashboard/models/model_lr_initial.pkl")
joblib.dump(nb, "submit_dashboard/models/model_nb_initial.pkl")
joblib.dump(svm, "submit_dashboard/models/model_svm_initial.pkl")

print("All models saved successfully.")