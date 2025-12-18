# store_processed.py
from spark_session import spark

# Load the merged cleaned dataset
merged_path = "hdfs://localhost:9000/processed/final_enriched_data"
df_merged = spark.read.parquet(merged_path)

# Final output path in HDFS
output_path = "hdfs://localhost:9000/processed/final_dataset"

# Store final processed dataset
df_merged.write.mode("overwrite").parquet(output_path)

print("Final processed dataset stored successfully!")
print(f"Saved to: {output_path}")
