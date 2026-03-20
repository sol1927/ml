# =========================================
# SENTIMENT ANALYSIS – STREAMLIT UI
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
    layout="wide"
)

# -------------------------------
# Load vectorizer & models
# -------------------------------
VECTORIZER_PATH = "submit_dashboard/models/tfidf_vectorizer.pkl"

MODELS = {
    "Linear SVM ⭐ (Best)": {
        "path": "submit_dashboard/models/model_svm_final.pkl",
        "accuracy": 92.32
    },
    "Logistic Regression": {
        "path": "submit_dashboard/models/model_lr_final.pkl",
        "accuracy": 91.59
    },
    "Naive Bayes": {
        "path": "submit_dashboard/models/model_nb_final.pkl",
        "accuracy": 72.12
    }
}

vectorizer = joblib.load(VECTORIZER_PATH)
models = {k: joblib.load(v["path"]) for k, v in MODELS.items()}

# -------------------------------
# Sentiment labels
# -------------------------------
INT_TO_LABEL = {
    -1: "Negative 😠",
     0: "Neutral 😐",
     1: "Positive 😊"
}

# -------------------------------
# Text cleaning
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|#[^\s]+|[^a-z\s]', ' ', text)
    return ' '.join(text.split())

# -------------------------------
# Batch prediction
# -------------------------------
def predict_batch(texts, model):
    X = vectorizer.transform([clean_text(t) for t in texts])
    preds = model.predict(X)
    labels = [INT_TO_LABEL[p] for p in preds]

    df = pd.DataFrame({"Text": texts, "Sentiment": labels})
    summary = df["Sentiment"].value_counts().to_frame("Count")
    summary["Percentage"] = (summary["Count"] / len(df) * 100).round(2)
    return df, summary

# ===============================
# UI HEADER
# ===============================
st.markdown(
    """
    <h2 style='text-align:center;'>💬 Social Media Sentiment Analysis</h2>
    <p style='text-align:center;color:gray;'>
    Analyze your social media text or comments
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ===============================
# MODEL SELECTION + ACCURACY
# ===============================
col1, col2 = st.columns([1, 2])

with col1:
    selected_model = st.selectbox(
        "🤖 Select Machine Learning Model",
        MODELS.keys()
    )

with col2:
    st.markdown("### 📊 Model Performance")
    perf_df = pd.DataFrame({
        "Model": MODELS.keys(),
        "Accuracy (%)": [v["accuracy"] for v in MODELS.values()]
    })
    st.dataframe(perf_df, use_container_width=True)

model = models[selected_model]

st.success(
    f"**{selected_model} selected — Accuracy: {MODELS[selected_model]['accuracy']}%**"
)

st.write("---")

# ===============================
# INPUT + OUTPUT (ONE PAGE FEEL)
# ===============================
left, right = st.columns(2)

# -------- INPUT --------
with left:
    st.subheader("📝 Input Text")
    input_method = st.radio("", ["Single Text", "Paste / Upload"], horizontal=True)

    user_text = ""
    uploaded_file = None

    if input_method == "Single Text":
        user_text = st.text_area("Enter text", height=120)
    else:
        user_text = st.text_area("Paste comments (one per line)", height=120)
        uploaded_file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])

    analyze = st.button("🚀 Analyze Sentiment", use_container_width=True)

# -------- OUTPUT --------
with right:
    st.subheader("📊 Results")

    if analyze:

        # ----- SINGLE TEXT -----
        if input_method == "Single Text" and user_text.strip():
            X = vectorizer.transform([clean_text(user_text)])
            pred = model.predict(X)[0]

            label = INT_TO_LABEL[pred]

            if pred == 1:
                st.success(label)
            elif pred == -1:
                st.error(label)
            else:
                st.info(label)

        # ----- PASTE TEXT -----
        elif input_method == "Paste / Upload" and user_text.strip():
            texts = [t.strip() for t in user_text.splitlines() if t.strip()]
            df, summary = predict_batch(texts, model)

            st.dataframe(df, height=220)
            st.dataframe(summary)

        # ----- FILE UPLOAD -----
        elif uploaded_file is not None:
            all_labels = []
            samples = []

            if uploaded_file.name.endswith(".txt"):
                texts = uploaded_file.read().decode("utf-8").splitlines()
                texts = [t for t in texts if t.strip()]
                df, summary = predict_batch(texts, model)

                st.dataframe(df.head(50), height=220)
                st.dataframe(summary)

            elif uploaded_file.name.endswith(".csv"):
                for chunk in pd.read_csv(uploaded_file, chunksize=100000):
                    if "text" not in chunk.columns:
                        st.error("CSV must contain a 'text' column")
                        st.stop()

                    texts = chunk["text"].astype(str).tolist()
                    df_chunk, _ = predict_batch(texts, model)

                    all_labels.extend(df_chunk["Sentiment"].tolist())

                    if len(samples) < 50:
                        samples.extend(df_chunk.head(50 - len(samples)).to_dict("records"))

                summary = pd.Series(all_labels).value_counts().to_frame("Count")
                summary["Percentage"] = (summary["Count"] / len(all_labels) * 100).round(2)

                st.dataframe(pd.DataFrame(samples), height=220)
                st.dataframe(summary)

        else:
            st.warning("Please enter text or upload a file.")

# ===============================
# FOOTER
# ===============================
st.write("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2026 Social Media Sentiment Project</p>",
    unsafe_allow_html=True
)