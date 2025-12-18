# =========================================
# Model Comparison Script - Corrected
# =========================================

import joblib
import pandas as pd
import numpy as np
from spark_session import spark  # Your SparkSession
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load dataset from HDFS
# -------------------------------
df_spark = spark.read.parquet("hdfs://localhost:9000/processed/final_dataset")

# Use a sample for faster evaluation
df = df_spark.limit(1000).toPandas()

# -------------------------------
# 2. Ensure required columns exist
# -------------------------------
if 'clean_text' not in df.columns or 'sentiment' not in df.columns:
    raise ValueError("Columns 'clean_text' or 'sentiment' not found in dataset")

# -------------------------------
# 3. Map sentiment strings to integers
# -------------------------------
sentiment_mapping = {"Negative": -1, "Neutral": 0, "Positive": 1}

# Convert sentiment column
df['sentiment'] = df['sentiment'].map(sentiment_mapping).fillna(0).astype(int)

# Check distribution
print("Sentiment distribution:\n", df['sentiment'].value_counts())

# -------------------------------
# 4. Load TF-IDF vectorizer & transform text
# -------------------------------
vectorizer = joblib.load("submit_dashboard/models/tfidf_vectorizer.pkl")
X = vectorizer.transform(df["clean_text"])
y = df["sentiment"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 5. Load trained models
# -------------------------------
lr = joblib.load("submit_dashboard/models/model_lr_initial.pkl")
nb = joblib.load("submit_dashboard/models/model_nb_initial.pkl")
svm = joblib.load("submit_dashboard/models/model_svm_initial.pkl")

models = {
    "Logistic Regression": lr,
    "Naive Bayes": nb,
    "Linear SVM": svm
}

# -------------------------------
# 6. Evaluate models
# -------------------------------
summary = []

# Define labels and plot labels
labels = [-1, 0, 1]  # sentiment encoding
plot_labels = ["Negative", "Neutral", "Positive"]

for name, model in models.items():
    print(f"\n=== {name} Evaluation ===")

    # Predict
    y_pred = model.predict(X_test)

    # If predictions are strings, map to integers
    if isinstance(y_pred[0], str):
        y_pred = pd.Series(y_pred).map(sentiment_mapping).fillna(0).astype(int)
    else:
        y_pred = pd.Series(y_pred).astype(int)

    y_test_int = y_test.astype(int)

    # Accuracy
    acc = accuracy_score(y_test_int, y_pred)
    print("Accuracy:", acc)

    # Classification report
    report_dict = classification_report(
        y_test_int, y_pred, labels=labels, target_names=plot_labels, output_dict=True, zero_division=0
    )
    print(classification_report(
        y_test_int, y_pred, labels=labels, target_names=plot_labels, zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test_int, y_pred, labels=labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=plot_labels, yticklabels=plot_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

    # Store summary
    summary.append({
        "Model": name,
        "Accuracy": acc,
        "Weighted F1": report_dict['weighted avg']['f1-score']
    })

# -------------------------------
# 7. Summary comparison plot
# -------------------------------
summary_df = pd.DataFrame(summary)

x = np.arange(len(summary_df['Model']))
width = 0.35

plt.figure(figsize=(8, 4))
plt.bar(x - width/2, summary_df['Accuracy'], width, label="Accuracy", alpha=0.8)
plt.bar(x + width/2, summary_df['Weighted F1'], width, label="Weighted F1", alpha=0.6)
plt.xticks(x, summary_df['Model'])
plt.ylabel("Score")
plt.title("Model Comparison (Test Set)")
plt.ylim(0, 1)
plt.legend()
plt.show()

# -------------------------------
# 8. Print summary
# -------------------------------
print("\n Model comparison completed. Summary:")
print(summary_df)