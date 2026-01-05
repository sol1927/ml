# ml_feature_pipeline_spark_vectorized.py
from spark_session import spark
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType
import pandas as pd
import joblib

# -------------------------------
# 1. Load dataset from HDFS
# -------------------------------
hdfs_input_path = "hdfs://localhost:9000/processed/final_dataset"
df_spark = spark.read.parquet(hdfs_input_path)
print("Loaded dataset from HDFS:", df_spark.count(), "rows")

# -------------------------------
# 2. Clean text (simple lowercase + trim)
# -------------------------------
from pyspark.sql.functions import lower, regexp_replace, trim

df_spark = df_spark.withColumn(
    "clean_text",
    trim(lower(regexp_replace(col("clean_text"), r'[^a-z\s]', "")))
)

# -------------------------------
# 3. Load vectorizer + model
# -------------------------------
VECTORIZER_PATH = "submit_dashboard/models/tfidf_vectorizer.pkl"
MODEL_PATH = "submit_dashboard/models/model_svm_final.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)
INT_TO_LABEL = {-1: "Negative", 0: "Neutral", 1: "Positive"}

# -------------------------------
# 4. Vectorized Pandas UDF for prediction
# -------------------------------
@pandas_udf(StringType())
def predict_sentiment_vectorized(text_series: pd.Series) -> pd.Series:
    X = vectorizer.transform(text_series)
    preds = model.predict(X)
    return pd.Series([INT_TO_LABEL[p] for p in preds])

# -------------------------------
# 5. Apply vectorized UDF
# -------------------------------
df_spark_pred = df_spark.withColumn("sentiment", predict_sentiment_vectorized(col("clean_text")))

# -------------------------------
# 6. Save results to HDFS
# -------------------------------
output_path = "hdfs://localhost:9000/processed/final_dataset_ml"
df_spark_pred.write.mode("overwrite").parquet(output_path)
print(f"ML-processed dataset saved to HDFS: {output_path}")

# -------------------------------
# 7. Optional: Summary
# -------------------------------
summary = df_spark_pred.groupBy("sentiment").count().toPandas()
summary["Percentage"] = (summary["count"] / summary["count"].sum() * 100).round(2)
print("\nSentiment Summary:\n", summary)
