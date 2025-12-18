# structured_clean.py
from spark_session import spark
from pyspark.sql.functions import col, trim, lower, regexp_replace

# Load structured cleaned data from previous step
input_path = "hdfs://localhost:9000/processed/structured_clean"
df_struct = spark.read.parquet(input_path)

# ============= BASIC CLEANING =============

# Remove special characters from text columns
for c in df_struct.columns:
    if df_struct.schema[c].dataType.simpleString() == "string":
        df_struct = df_struct.withColumn(c, regexp_replace(col(c), r'[^a-zA-Z0-9\s]', ''))

# Drop duplicates
df_struct = df_struct.dropDuplicates()

print("=== Structured Cleaned Preview ===")
df_struct.show(5, truncate=False)