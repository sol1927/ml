# =========================================
# Full Model Evaluation Script
# =========================================

import pandas as pd
from spark_session import spark
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import os

# -------------------------------
# 1. Load dataset from HDFS
# -------------------------------
df_spark = spark.read.parquet("hdfs://localhost:9000/processed/final_dataset")
df = df_spark.limit(1000).toPandas()  # sample for faster evaluation

# -------------------------------
# 2. Ensure 'clean_text' exists
# -------------------------------
if 'clean_text' not in df.columns:
    raise ValueError("Column 'clean_text' not found in dataset")

# -------------------------------
# 3. Ensure sentiment column exists
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

# Force integer type for consistency
def map_sentiment(x):
    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        return int(x)
    x_str = str(x).lower()
    if x_str in ["positive", "1"]:
        return 1
    elif x_str in ["negative", "-1"]:
        return -1
    else:
        return 0

df['sentiment'] = df['sentiment'].apply(map_sentiment)

# Check distribution
print("Sentiment distribution:\n", df['sentiment'].value_counts())

# -------------------------------
# 4. Load TF-IDF vectorizer
# -------------------------------
vectorizer = joblib.load("submit_dashboard/models/tfidf_vectorizer.pkl")
X = vectorizer.transform(df["clean_text"])
y = df["sentiment"]

# -------------------------------
# 5. Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. Load saved models
# -------------------------------
models = {
    "Logistic Regression": joblib.load("submit_dashboard/models/model_lr_initial.pkl"),
    "Naive Bayes": joblib.load("submit_dashboard/models/model_nb_initial.pkl"),
    "Linear SVM": joblib.load("submit_dashboard/models/model_svm_initial.pkl")
}

# Labels for evaluation
labels = [-1, 0, 1]
label_names = ["Negative", "Neutral", "Positive"]

# -------------------------------
# 7. Evaluate models
# -------------------------------
for name, model in models.items():
    print(f"\n=== {name} Evaluation ===")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Ensure integer type for predictions
    y_pred = pd.Series(y_pred).apply(map_sentiment)
    y_true = y_test.apply(map_sentiment)
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=label_names,
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {name}")
    plt.show()
