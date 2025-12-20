# =========================================
# SENTIMENT ANALYSIS – STREAMLIT UI
# Big Data Group Assignment – Group F
# =========================================

import streamlit as st
import joblib

# -------------------------------
# Page configuration (MUST be first)
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
# UI Header
# -------------------------------
st.markdown(
    """
    <h4 style='text-align: center; color: gray;'>
        Big Data Group Assignment
    </h4>
    <h1 style='text-align: center;'>
        Social Media Sentiment
    </h1>
    <p style='text-align: center; font-size: 18px;'>
        Enter text below and click the button to predict sentiment
    </p>
    """,
    unsafe_allow_html=True
)

st.write("---")

# -------------------------------
# Text input
# -------------------------------
user_text = st.text_area(
    "📝 Enter your text:",
    height=100,
    placeholder="Type a social media comment or post here..."
)

# -------------------------------
# Predict button
# -------------------------------
if st.button("Click here"):

    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Vectorize input
        X = vectorizer.transform([user_text])

        # Predict sentiment
        prediction = model.predict(X)[0]
        label = INT_TO_LABEL[prediction]

        # Display result
        st.write("---")
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(label)
        elif prediction == -1:
            st.error(label)
        else:
            st.info(label)

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.markdown(
    "<h5 style='text-align: center;'>Group F • Big Data Analytics Project</h5>",
    unsafe_allow_html=True
)