# semi_clean.py
from spark_session import spark
from pyspark.sql.functions import col, trim, lower, regexp_replace

# Load semi-structured cleaned data
input_path = "hdfs://localhost:9000/processed/semi_clean"
df_semi = spark.read.parquet(input_path)

# ============= BASIC CLEANING =============

# Remove special characters
for c in df_semi.columns:
    if df_semi.schema[c].dataType.simpleString() == "string":
        df_semi = df_semi.withColumn(c, regexp_replace(col(c), r'[^a-zA-Z0-9\s]', ''))

# Drop duplicates
df_semi = df_semi.dropDuplicates()

print("=== Semi-Structured Cleaned Preview ===")
df_semi.show(5, truncate=False)