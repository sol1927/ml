# =========================================
# FINAL ML PIPELINE – INTERACTIVE SENTIMENT PREDICTION
# =========================================

import joblib
import pandas as pd
from IPython.display import clear_output  # for clearing previous output in Jupyter

# -------------------------------
# 1. Load vectorizer & best model
# -------------------------------
VECTORIZER_PATH = "submit_dashboard/models/tfidf_vectorizer.pkl"
MODEL_PATH = "submit_dashboard/models/model_svm_final.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

print(" Vectorizer & model loaded successfully!\n")

# -------------------------------
# 2. Sentiment label mappings
# -------------------------------
INT_TO_LABEL = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

# -------------------------------
# 3. Prediction function
# -------------------------------
def predict_sentiment(texts):
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    
    pred_labels = [INT_TO_LABEL[p] for p in predictions]
    
    return pd.DataFrame({
        "Input": texts,
        "Prediction": pred_labels,
        "Prediction (int)": predictions
    })

# -------------------------------
# 4. Interactive loop
# -------------------------------
print("Type your text to predict sentiment. Type 'exit' to quit.\n")

while True:
    user_input = input("Your text: ")
    
    # Clear previous output every time
    clear_output(wait=True)
    
    if user_input.lower() == "exit":
        print("Exiting sentiment prediction. Goodbye!")
        break
    elif user_input.strip() == "":
        print("Type your text to predict sentiment. Type 'exit' to quit.\n")
        print(" Empty input, please type something.\n")
        continue

    # Show current prediction only
    result = predict_sentiment([user_input])
    print("Type your text to predict sentiment. Type 'exit' to quit.\n")
    print(f"Input: {result['Input'][0]}")
    print(f"Prediction: {result['Prediction'][0]} ({result['Prediction (int)'][0]})\n")