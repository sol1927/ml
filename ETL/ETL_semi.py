from spark_session import spark
from pyspark.sql.functions import trim, lower, col, lit

# ================================
# 1. LOAD FROM HDFS (Extract)
# ================================
semi_path = "hdfs://localhost:9000/raw/semi"
df_semi = spark.read.parquet(semi_path)

print("=== Semi-Structured Raw Sample ===")
df_semi.show(5, truncate=False)

# ================================
# 2. TRANSFORM
# ================================

# Clean whitespace + lowercase all string columns
for c in df_semi.columns:
    if df_semi.schema[c].dataType.simpleString() == "string":
        df_semi = df_semi.withColumn(c, lower(trim(col(c))))

# Rename 'body' to 'text'
if "body" in df_semi.columns:
    df_semi = df_semi.withColumnRenamed("body", "text")

# Add source column
df_semi = df_semi.withColumn("source", lit("semi_structured"))

print("=== Semi-Structured After ETL ===")
df_semi.show(5, truncate=False)

# ================================
# 3. LOAD BACK TO HDFS
# ================================
output_semi = "hdfs://localhost:9000/processed/semi_clean"

df_semi.write.mode("overwrite").parquet(output_semi)

print("Semi-Structured ETL Pipeline Complete!")
