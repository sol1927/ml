# eda_load.py
from spark_session import spark

# Load final merged dataset from HDFS
input_path = "hdfs://localhost:9000/processed/final_dataset"
df = spark.read.parquet(input_path)

print("=== Dataset Schema ===")
df.printSchema()

print("=== Sample Rows ===")
df.show(5, truncate=False)

# Row count
print("Total Rows:", df.count())