# merge_cleaned.py
from spark_session import spark
from pyspark.sql.functions import lit

# Load cleaned structured data
struct_path = "hdfs://localhost:9000/processed/structured_clean"
df_struct = spark.read.parquet(struct_path)

# Load cleaned semi-structured data
semi_path = "hdfs://localhost:9000/processed/semi_clean_final"
df_semi = spark.read.parquet(semi_path)

# ============= MERGING LOGIC =============

# Find all unique columns between both DataFrames
all_cols = list(set(df_struct.columns) | set(df_semi.columns))

# Function to align columns
def align_columns(df, all_cols):
    for c in all_cols:
        if c not in df.columns:
            df = df.withColumn(c, lit(None))
    return df.select(all_cols)

# Align
df_struct = align_columns(df_struct, all_cols)
df_semi = align_columns(df_semi, all_cols)

# Merge (Union)
df_merged = df_struct.unionByName(df_semi)

print("=== Merged Cleaned Dataset Preview ===")
df_merged.show(10, truncate=False)
print("Total rows:", df_merged.count())

# ============= SAVE MERGED DATASET =============
output_path = "hdfs://localhost:9000/processed/merged_temp"

df_merged.write.mode("overwrite").parquet(output_path)

print(f"\nMerged dataset successfully saved to: {output_path}")