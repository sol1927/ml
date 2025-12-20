# eda_visuals.py
from spark_session import spark
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
input_path = "hdfs://localhost:9000/processed/final_dataset"
df_spark = spark.read.parquet(input_path)

# Convert to Pandas for visualization (small dataset)
df = df_spark.toPandas()

# =======================
# Example 1: Top text values bar chart
# =======================
text_cols = df.select_dtypes(include='object').columns
if len(text_cols) > 0:
    col_name = text_cols[0]
    top_values = df[col_name].value_counts().head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_values.values, y=top_values.index, palette="viridis")
    plt.title(f"Top 10 values in '{col_name}'")
    plt.xlabel("Count")
    plt.ylabel(col_name)
    plt.show()

# =======================
# Example 2: Numeric distribution
# =======================
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col_name in numeric_cols[:5]:  # first 5 numeric columns
    plt.figure(figsize=(8,4))
    sns.histplot(df[col_name].dropna(), kde=True, color="skyblue")
    plt.title(f"Distribution of '{col_name}'")
    plt.show()