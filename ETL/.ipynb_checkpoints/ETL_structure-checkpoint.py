from spark_session import spark
from pyspark.sql.functions import trim, lower, col, lit

# ================================
# 1. LOAD FROM HDFS (Extract)
# ================================
structured_path = "hdfs://localhost:9000/raw/structured"
df_struct = spark.read.parquet(structured_path)

print("=== Structured Raw Sample ===")
df_struct.show(5, truncate=False)

# ================================
# 2. TRANSFORM
# ================================

# Clean whitespace + lowercase all string columns
for c in df_struct.columns:
    if df_struct.schema[c].dataType.simpleString() == "string":
        df_struct = df_struct.withColumn(c, lower(trim(col(c))))

# Add source column
df_struct = df_struct.withColumn("source", lit("structured"))

print("=== Structured After ETL ===")
df_struct.show(5, truncate=False)

# ================================
# 3. LOAD BACK TO HDFS
# ================================
output_struct = "hdfs://localhost:9000/processed/structured_clean"

df_struct.write.mode("overwrite").parquet(output_struct)

print("Structured ETL Pipeline Complete!")