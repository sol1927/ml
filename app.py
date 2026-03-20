# =========================================
# SENTIMENT ANALYSIS – STREAMLIT UI
# =========================================

import streamlit as st
import joblib

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Social Media Sentiment",
    page_icon="💬",
    layout="centered"
)

# -------------------------------
# Load vectorizer & model with caching
# -------------------------------
VECTORIZER_PATH = "submit_dashboard/models/tfidf_vectorizer.pkl"
MODEL_PATH = "submit_dashboard/models/model_svm_final.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

vectorizer = load_model(VECTORIZER_PATH)
model = load_model(MODEL_PATH)

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
    <h1 style='text-align: center;'>
        Social Media Sentiment Analysis
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
if st.button("Predict Sentiment"):

    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            # Vectorize input
            X = vectorizer.transform([user_text])

            # Predict sentiment
            prediction = model.predict(X)[0]
            label = INT_TO_LABEL.get(prediction, "Unknown")

            # Display result
            st.write("---")
            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.success(label)
            elif prediction == -1:
                st.error(label)
            else:
                st.info(label)

        except Exception as e:
            st.error(f"Error predicting sentiment: {e}")

# -------------------------------
# Footer
# -------------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center; font-size:14px;'>© 2026 Social Media Sentiment Project</p>",
    unsafe_allow_html=True
)