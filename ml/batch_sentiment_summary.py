import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# 1. Load vectorizer & trained model
# -------------------------------
VECTORIZER_PATH = "submit_dashboard/models/tfidf_vectorizer.pkl"
MODEL_PATH = "submit_dashboard/models/model_svm_final.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

INT_TO_LABEL = {-1: "Negative", 0: "Neutral", 1: "Positive"}

# -------------------------------
# 2. Prediction function
# -------------------------------
def predict_batch(texts):
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    labels = [INT_TO_LABEL[p] for p in preds]
    
    df = pd.DataFrame({
        "Text": texts,
        "Sentiment": labels
    })
    
    # Calculate counts & percentages
    summary = df["Sentiment"].value_counts().to_frame("Count")
    summary["Percentage"] = (summary["Count"] / len(df) * 100).round(2)
    
    return df, summary

# -------------------------------
# 3. Interactive batch input
# -------------------------------
print("Paste your comments (type 'END' on a new line to finish):\n")

lines = []
while True:
    line = input()
    if line.strip().upper() == "END":
        break
    if line.strip() != "":
        lines.append(line.strip())

if not lines:
    print("No comments entered. Exiting.")
else:
    labeled_df, summary_df = predict_batch(lines)
    
    print("\n=== Sentiment Categorization ===")
    print(labeled_df)
    
    print("\n=== Summary (Counts & Percentages) ===")
    print(summary_df)