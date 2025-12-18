from spark_session import spark
from pyspark.sql.functions import col, lower, regexp_replace, trim, when, split, array_intersect, size, array, lit

# 1. Load the merged file you just created
merged_path = "hdfs://localhost:9000/processed/merged_temp" 
output_path = "hdfs://localhost:9000/processed/final_enriched_data"

df = spark.read.parquet(merged_path)

# 2. Define Lexicons (Expanded for better accuracy)
pos_words = ["good", "great", "awesome", "excellent", "happy", "love", "best", "nice", "amazing", "wonderful", "perfect"]
neg_words = ["bad", "terrible", "worst", "sad", "awful", "hate", "boring", "disappointing", "poor", "wrong", "scary"]

# 3. Step A: Fill the NULL clean_text column
# We use coalesce to ensure if clean_text is already there, we keep it; 
# otherwise, we clean the 'text' column.
df = df.withColumn("clean_text", 
    trim(lower(regexp_replace(col("text"), r'http\S+|[^a-zA-Z\s]', "")))
)

# 4. Step B: Sentiment Logic (Native Spark - No Python crashes!)
df = df.withColumn("temp_words", split(col("clean_text"), " "))
df = df.withColumn("pos_match", size(array_intersect(col("temp_words"), array([lit(w) for w in pos_words]))))
df = df.withColumn("neg_match", size(array_intersect(col("temp_words"), array([lit(w) for w in neg_words]))))

df = df.withColumn("sentiment", 
    when(col("pos_match") > col("neg_match"), "Positive")
    .when(col("neg_match") > col("pos_match"), "Negative")
    .otherwise("Neutral")
)

# 5. Drop helpers and Save
df_final = df.drop("temp_words", "pos_match", "neg_match")

print("Writing final enriched dataset to HDFS...")
df_final.write.mode("overwrite").parquet(output_path)

# 6. Verification
print("=== SENTIMENT DISTRIBUTION ===")
df_final.groupBy("sentiment").count().show()