# eda_text_top.py
from spark_session import spark
from pyspark.sql.functions import col, desc

# Load dataset
input_path = "hdfs://localhost:9000/processed/final_dataset"
df = spark.read.parquet(input_path)

# Detect text columns
text_cols = [c for c, t in df.dtypes if t == 'string']

if text_cols:
    first_text = text_cols[0]
    print(f"=== Most Frequent Values in '{first_text}' ===")
    df.groupBy(first_text).count().orderBy(desc("count")).show(10, truncate=False)
else:
    print("No text columns found.")
