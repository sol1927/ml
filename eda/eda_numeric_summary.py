# eda_numeric_summary.py
from spark_session import spark

# Load dataset
input_path = "hdfs://localhost:9000/processed/final_dataset"
df = spark.read.parquet(input_path)

# Detect numeric columns
numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float")]

if numeric_cols:
    print("=== Numeric Columns Summary ===")
    df.select(numeric_cols).describe().show()
else:
    print("No numeric columns found.")
