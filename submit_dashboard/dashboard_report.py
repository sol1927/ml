# =========================================
# Dashboard Draft / EDA Script
# =========================================

from spark_session import spark
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib

# -------------------------------
# FIX 1: Font that supports emojis
# -------------------------------
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

sns.set(style="whitegrid")  # pretty plots

# -------------------------------
# 1. Load merged dataset
# -------------------------------
hdfs_path = "hdfs://localhost:9000/processed/final_dataset"
df_spark = spark.read.parquet(hdfs_path)
df = df_spark.toPandas()

# -------------------------------
# 2. Prepare output folder
# -------------------------------
output_dir = "submit_dashboard/plots"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 3. Basic info
# -------------------------------
print("=== Basic Info ===")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Column names:", df.columns.tolist())
print("\nMissing values per column:")
print(df.isnull().sum())

# -------------------------------
# 4. Numeric summary
# -------------------------------
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if numeric_cols:
    print("\n=== Numeric Summary ===")
    print(df[numeric_cols].describe())

# -------------------------------
# 5. Categorical / Text summary
# -------------------------------
categorical_cols = df.select_dtypes(include='object').columns.tolist()
if categorical_cols:
    print("\n=== Categorical Summary ===")
    for col in categorical_cols:
        print(f"\nTop 5 values in {col}:")
        top_values = df[col].value_counts().head(5)
        if top_values.empty:
            print("No data available")
        else:
            print(top_values)

# -------------------------------
# 6. Bar charts for categorical columns
# -------------------------------
for col in categorical_cols:
    counts = df[col].value_counts().head(10)
    if counts.empty:
        print(f"Skipping {col}: no data to plot")
        continue

    plt.figure(figsize=(8, 4))
    counts.plot(kind='bar')
    plt.title(f"Top 10 {col} values")
    plt.ylabel("Count")
    plt.xlabel(col)
    plt.xticks(rotation=45)

    # FIX 2: Prevent tight_layout warnings
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(output_dir, f"{col}_bar.png")
    plt.savefig(save_path)
    print(f"Saved bar chart: {save_path}")
    plt.show()

# -------------------------------
# 7. Histograms for numeric columns
# -------------------------------
for col in numeric_cols:
    if df[col].dropna().empty:
        print(f"Skipping {col}: no data to plot")
        continue

    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")

    # FIX 2 again
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(output_dir, f"{col}_hist.png")
    plt.savefig(save_path)
    print(f"Saved histogram: {save_path}")
    plt.show()

# -------------------------------
# 8. Correlation heatmap
# -------------------------------
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 6))
    corr = df[numeric_cols].corr()

    if not corr.empty:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")

        # FIX 2 again
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(save_path)
        print(f"Saved correlation heatmap: {save_path}")
        plt.show()
