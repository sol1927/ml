# =========================================
# Corrected ML Model Evaluation Script
# =========================================

import pandas as pd
from spark_session import spark
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -------------------------------
# 1. Load full dataset from HDFS
# -------------------------------
df_spark = spark.read.parquet("hdfs://localhost:9000/processed/final_dataset")
df = df_spark.toPandas()

# -------------------------------
# 2. Handle missing clean_text
# -------------------------------
if 'clean_text' not in df.columns:
    raise ValueError("Column 'clean_text' not found")
df['clean_text'] = df['clean_text'].fillna("")

# -------------------------------
# 3. Generate sentiment labels if missing
# -------------------------------
from textblob import TextBlob
if 'sentiment' not in df.columns or df['sentiment'].isnull().all():
    def get_sentiment(text):
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0.1:
            return 1
        elif polarity < -0.1:
            return -1
        else:
            return 0
    df['sentiment'] = df['clean_text'].apply(get_sentiment)

# -------------------------------
# 4. Balance dataset exactly like training
# -------------------------------
count_neg = len(df[df.sentiment == -1])
count_neu = len(df[df.sentiment == 0])
count_pos = len(df[df.sentiment == 1])
min_count = min(count_neg, count_neu, count_pos)

df_neg = df[df.sentiment == -1].sample(min_count, random_state=42)
df_neu = df[df.sentiment == 0].sample(min_count, random_state=42)
df_pos = df[df.sentiment == 1].sample(min_count, random_state=42)

df_balanced = pd.concat([df_neg, df_neu, df_pos]).sample(frac=1, random_state=42)
print("Balanced distribution for evaluation:\n", df_balanced['sentiment'].value_counts())

# -------------------------------
# 5. Load TF-IDF vectorizer
# -------------------------------
vectorizer_path = "submit_dashboard/models/tfidf_vectorizer.pkl"
if not os.path.exists(vectorizer_path):
    raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")
vectorizer = joblib.load(vectorizer_path)

X = vectorizer.transform(df_balanced["clean_text"])
y = df_balanced["sentiment"]

# -------------------------------
# 6. Train/Test split
# -------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# 7. Load trained models
# -------------------------------
models = {
    "Logistic Regression": joblib.load("submit_dashboard/models/model_lr_final.pkl"),
    "Naive Bayes": joblib.load("submit_dashboard/models/model_nb_final.pkl"),
    "Linear SVM": joblib.load("submit_dashboard/models/model_svm_final.pkl")
}

# -------------------------------
# 8. Evaluate models
# -------------------------------
labels = [-1, 0, 1]
label_names = ["Negative", "Neutral", "Positive"]

for name, model in models.items():
    print(f"\n=== {name} Evaluation ===")
    
    y_pred = model.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=label_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {name}")
    plt.show()