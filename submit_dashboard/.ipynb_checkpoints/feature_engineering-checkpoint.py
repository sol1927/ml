import pandas as pd
from spark_session import spark
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from textblob import TextBlob

# -------------------------------
# Download stopwords (safe check)
# -------------------------------
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# -------------------------------
# Step 2a: Load dataset
# -------------------------------
hdfs_path = "hdfs://localhost:9000/processed/final_dataset"
df_spark = spark.read.parquet(hdfs_path)

# Use sample to avoid memory issues (remove limit for full run)
df = df_spark.limit(1000).toPandas()

print("Dataset loaded. Preview:")
print(df.head())

# -------------------------------
# Step 2b: Clean text
# -------------------------------
def clean_text_fn(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(w for w in text.split() if w not in stop_words)
    return text.strip()

# IMPORTANT: reuse existing clean_text column safely
df['clean_text'] = df['clean_text'].fillna('').apply(clean_text_fn)

print("Text cleaned. Preview:")
print(df[['clean_text']].head())

# -------------------------------
# Step 2c: TF-IDF feature extraction
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2
)

X = vectorizer.fit_transform(df['clean_text'])

print("TF-IDF features created. Shape:", X.shape)

# -------------------------------
# Step 2d: Sentiment labeling
# -------------------------------
def sentiment_label(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 1
    elif polarity < 0:
        return -1
    else:
        return 0

df['sentiment'] = df['clean_text'].apply(sentiment_label)
y = df['sentiment']

print("Sentiment labels created. Preview:")
print(df[['clean_text', 'sentiment']].head())

# -------------------------------
# Step 2e: Save TF-IDF vectorizer
# -------------------------------
os.makedirs('submit_dashboard/models', exist_ok=True)
joblib.dump(vectorizer, 'submit_dashboard/models/tfidf_vectorizer.pkl')

print("TF-IDF vectorizer saved successfully!")