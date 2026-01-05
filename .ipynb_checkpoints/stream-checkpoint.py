# streamlit_sentiment.py

import streamlit as st
import pandas as pd
import joblib
import re

# -------------------------------
# 1. Load vectorizer & trained model
# -------------------------------
VECTORIZER_PATH = "submit_dashboard/models/tfidf_vectorizer.pkl"
MODEL_PATH = "submit_dashboard/models/model_svm_final.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

INT_TO_LABEL = {-1: "Negative", 0: "Neutral", 1: "Positive"}

# -------------------------------
# 2. Clean text function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|#[^\s]+|[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

# -------------------------------
# 3. Prediction function
# -------------------------------
def predict_sentiment_batch(texts):
    texts_clean = [clean_text(t) for t in texts]
    X = vectorizer.transform(texts_clean)
    preds = model.predict(X)
    labels = [INT_TO_LABEL[p] for p in preds]

    df = pd.DataFrame({
        "Text": texts,
        "Sentiment": labels
    })

    summary = df["Sentiment"].value_counts().to_frame("Count")
    summary["Percentage"] = (summary["Count"] / len(df) * 100).round(2)

    return df, summary

# -------------------------------
# 4. Streamlit App UI
# -------------------------------
st.title("Batch Sentiment Analysis")
st.write("Analyze sentiment from pasted text, CSV files, or TXT files.")

input_method = st.radio(
    "Select input method:",
    ("Paste Text", "Upload File")
)

# -------------------------------
# PASTE TEXT
# -------------------------------
if input_method == "Paste Text":
    user_input = st.text_area("Paste comments (one per line)")

    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please paste some text.")
        else:
            texts = [line.strip() for line in user_input.splitlines() if line.strip()]
            df_labeled, df_summary = predict_sentiment_batch(texts)

            st.subheader("Sentiment Categorization")
            st.dataframe(df_labeled)

            st.subheader("Summary")
            st.dataframe(df_summary)

# -------------------------------
# UPLOAD FILE (CSV or TXT)
# -------------------------------
elif input_method == "Upload File":
    uploaded_file = st.file_uploader(
        "Upload a CSV or TXT file",
        type=["csv", "txt"]
    )

    if uploaded_file is not None:

        all_labels = []

        # ---------- TXT FILE ----------
        if uploaded_file.name.endswith(".txt"):
            texts = uploaded_file.read().decode("utf-8").splitlines()
            texts = [t for t in texts if t.strip()]

            df_labeled, df_summary = predict_sentiment_batch(texts)

            st.subheader("Sentiment Categorization")
            st.dataframe(df_labeled)

            st.subheader("Summary")
            st.dataframe(df_summary)

        # ---------- CSV FILE ----------
        elif uploaded_file.name.endswith(".csv"):
            chunksize = 100000

            for chunk in pd.read_csv(uploaded_file, chunksize=chunksize):
                if "text" not in chunk.columns:
                    st.error("CSV must contain a 'text' column")
                    st.stop()

                texts = chunk["text"].astype(str).tolist()
                df_labeled, _ = predict_sentiment_batch(texts)
                all_labels.extend(df_labeled["Sentiment"].tolist())

            summary = pd.Series(all_labels).value_counts().to_frame("Count")
            summary["Percentage"] = (summary["Count"] / len(all_labels) * 100).round(2)

            st.subheader("Summary for Large Dataset")
            st.dataframe(summary)
