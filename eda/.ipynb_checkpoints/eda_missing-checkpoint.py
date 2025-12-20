# eda_missing.py
from spark_session import spark
from pyspark.sql.functions import col, count, when

# Load dataset
input_path = "hdfs://localhost:9000/processed/final_dataset"
df = spark.read.parquet(input_path)

# Count missing or empty values per column
missing_values = df.select([
    count(when(col(c).isNull() | (col(c) == ""), c)).alias(c)
    for c in df.columns
])

print("=== Missing Values per Column ===")
missing_values.show(truncate=False)
