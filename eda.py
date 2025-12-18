# eda.py
from spark_session import spark
from pyspark.sql.functions import col, count, isnan, when, desc

# =======================
# 1. LOAD MERGED DATASET
# =======================
input_path = "hdfs://localhost:9000/processed/merged_temp"
df = spark.read.parquet(input_path)

print("\n=== Dataset Schema ===")
df.printSchema()

print("\n=== Sample Rows ===")
df.show(10, truncate=False)

# =======================
# 2. ROW COUNT
# =======================
print("\n=== Total Rows ===")
print(df.count())

# =======================
# 3. NULL ANALYSIS (SAFE)
# =======================
print("\n=== Missing Values per Column ===")

numeric_cols = [c for c, t in df.dtypes if t in ("double", "float")]
null_counts = []

for c in df.columns:
    if c in numeric_cols:
        # numeric columns: check null or NaN
        null_counts.append(count(when(col(c).isNull() | isnan(col(c)), c)).alias(c))
    else:
        # non-numeric columns: check null only
        null_counts.append(count(when(col(c).isNull(), c)).alias(c))

df.select(null_counts).show(truncate=False)

# =======================
# 4. TOP VALUES FOR TEXT COLUMN
# =======================
text_cols = [c for c, t in df.dtypes if t == 'string']

if text_cols:
    first_text = text_cols[0]
    print(f"\n=== Most Frequent Values in '{first_text}' ===")
    df.groupBy(first_text).count().orderBy(desc("count")).show(10, truncate=False)
else:
    print("\nNo text columns found.")

# =======================
# 5. BASIC STATS FOR NUMERIC COLUMNS
# =======================
numeric_cols = [c for c, t in df.dtypes if t in ("int", "bigint", "double", "float")]

if numeric_cols:
    print("\n=== Numeric Summary ===")
    df.describe(numeric_cols).show()
else:
    print("\nNo numeric columns found.")