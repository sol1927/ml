from spark_session import spark

# Path
input_path = "file:///C:/Users/hp/Desktop/new/semi_data"
output_path = "hdfs://localhost:9000/raw/semi"

# Read JSON
df = spark.read.json(input_path)

# Preview
df.show(5, truncate=False)

# Write to HDFS
df.write.mode("overwrite").parquet(output_path)

print("Semi-structured ingestion complete!")