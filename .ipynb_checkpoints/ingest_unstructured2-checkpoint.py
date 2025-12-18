from spark_session import spark
from pyspark.sql.functions import monotonically_increasing_id

input_path = "file:///C:/Users/hp/Desktop/new/unst_reddit_comments.xlsx"
output_path = "hdfs://localhost:9000/raw/unstructured2"

# Read raw text
rdd = spark.sparkContext.textFile(input_path)

# Convert to dataframe
df = rdd.map(lambda x: (x,)).toDF(["raw_line"])

df = df.withColumn("id", monotonically_increasing_id())

df.show(5, truncate=False)

# Write to HDFS
df.write.mode("overwrite").parquet(output_path)

print("Unstructured text file #2 ingestion complete!")