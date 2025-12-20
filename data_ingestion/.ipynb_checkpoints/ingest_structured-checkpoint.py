from spark_session import spark

# Path
input_path = "file:///C:/Users/hp/Desktop/new/stru_covid19_tweets.csv"
output_path = "hdfs://localhost:9000/raw/structured"

# Read CSV
df = spark.read.csv(input_path, header=True, inferSchema=True)

# Preview
df.show(5, truncate=False)

# Write to HDFS
df.write.mode("overwrite").parquet(output_path)

print("Structured ingestion complete!")
