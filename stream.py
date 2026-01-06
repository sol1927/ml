# =========================================
# SENTIMENT ANALYSIS – STREAMLIT UI
# Big Data Group Assignment – Group F
# =========================================

import streamlit as st
import pandas as pd
import joblib
import re

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Social Media Sentiment",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# Load vectorizer & model
# -------------------------------
VECTORIZER_PATH = "submit_dashboard/models/tfidf_vectorizer.pkl"
MODEL_PATH = "submit_dashboard/models/model_svm_final.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

# -------------------------------
# Sentiment label mapping
# -------------------------------
INT_TO_LABEL = {
    -1: "Negative 😠",
     0: "Neutral 😐",
     1: "Positive 😊"
}

# -------------------------------
# Text preprocessing function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|#[^\s]+|[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text

# -------------------------------
# Prediction function
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
# UI Header
# -------------------------------
st.markdown(
    """
    <h4 style='text-align: center; color: gray;'>Big Data Group Assignment</h4>
    <h1 style='text-align: center;'>Social Media Sentiment</h1>
    <p style='text-align: center; font-size: 18px;'>
        Enter text below or upload a file to predict sentiment
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---")

# -------------------------------
# Input method selection
# -------------------------------
input_method = st.radio(
    "Select input method:",
    ("Single Text", "Paste Text / Upload File")
)

# -------------------------------
# SINGLE TEXT (app.py style)
# -------------------------------
if input_method == "Single Text":
    user_text = st.text_area(
        "📝 Enter your text:",
        height=100,
        placeholder="Type a social media comment or post here..."
    )

    if st.button("Predict Sentiment"):
        if user_text.strip() == "":
            st.warning("⚠️ Please enter some text.")
        else:
            X = vectorizer.transform([user_text])
            prediction = model.predict(X)[0]
            label = INT_TO_LABEL[prediction]

            st.write("---")
            st.subheader("📊 Prediction Result")
            if prediction == 1:
                st.success(label)
            elif prediction == -1:
                st.error(label)
            else:
                st.info(label)

# -------------------------------
# BATCH TEXT / FILE UPLOAD (stream.py style)
# -------------------------------
elif input_method == "Paste Text / Upload File":
    user_input = st.text_area("Paste comments (one per line)")

    uploaded_file = st.file_uploader(
        "Or upload a CSV or TXT file",
        type=["csv", "txt"]
    )

    if st.button("Analyze Batch"):

        all_labels = []

        # ---------- Paste Text ----------
        if user_input.strip():
            texts = [line.strip() for line in user_input.splitlines() if line.strip()]
            df_labeled, df_summary = predict_sentiment_batch(texts)

            st.subheader("Sentiment Categorization")
            st.dataframe(df_labeled)

            st.subheader("Summary")
            st.dataframe(df_summary)

        # ---------- Uploaded File ----------
        elif uploaded_file is not None:

            # TXT File
            if uploaded_file.name.endswith(".txt"):
                texts = uploaded_file.read().decode("utf-8").splitlines()
                texts = [t for t in texts if t.strip()]

                df_labeled, df_summary = predict_sentiment_batch(texts)

                st.subheader("Sentiment Categorization")
                st.dataframe(df_labeled)

                st.subheader("Summary")
                st.dataframe(df_summary)

            # CSV File
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

        else:
            st.warning("⚠️ Please paste text or upload a file for analysis.")

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.markdown(
    "<h5 style='text-align: center;'>Group F • Big Data Analytics Project</h5>",
    unsafe_allow_html=True
)
