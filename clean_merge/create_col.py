from spark_session import spark
from pyspark.sql.functions import col, lower, regexp_replace, trim, when, split, array_intersect, size, array, lit

# ---------------------------------------------------------
# 1. DEFINE POSITIVE/NEGATIVE WORD LISTS (Lexicon)
# ---------------------------------------------------------
# You can expand these lists to be more accurate
pos_words = ["good", "great", "awesome", "excellent", "happy", "positive", "love", "best", "cool", "nice"]
neg_words = ["bad", "terrible", "worst", "sad", "awful", "negative", "hate", "boring", "disappointing", "poor"]

# ---------------------------------------------------------
# 2. PROCESS SEMI-STRUCTURED DATA
# ---------------------------------------------------------
print("Starting High-Speed Sentiment Processing...")
semi_path = "hdfs://localhost:9000/processed/semi_clean"
semi_path_final = "hdfs://localhost:9000/processed/semi_clean_final"

# Read and Repartition heavily to spread the load
df = spark.read.parquet(semi_path).repartition(200)

# A. Clean Text (Remove non-alphabets)
df = df.withColumn("clean_text", 
    trim(lower(regexp_replace(col("text"), r'http\S+|[^a-zA-Z\s]', "")))
)

# B. Calculate Sentiment using Word Intersections (Native Spark - No UDF)
# 1. Convert text to array of words
df = df.withColumn("word_array", split(col("clean_text"), " "))

# 2. Count matches for positive and negative words
df = df.withColumn("pos_count", size(array_intersect(col("word_array"), array([lit(w) for w in pos_words]))))
df = df.withColumn("neg_count", size(array_intersect(col("word_array"), array([lit(w) for w in neg_words]))))

# 3. Determine Sentiment Label
df = df.withColumn("sentiment", 
    when(col("pos_count") > col("neg_count"), "Positive")
    .when(col("neg_count") > col("pos_count"), "Negative")
    .otherwise("Neutral")
)

# C. Cleanup and Save
# Remove temporary helper columns before saving
df_final = df.drop("word_array", "pos_count", "neg_count")

print("Writing results to HDFS...")
df_final.write.mode("overwrite").parquet(semi_path_final)
print(f"SUCCESS: Processed data saved to {semi_path_final}")