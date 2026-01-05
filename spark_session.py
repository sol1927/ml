from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("Bigdata") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark session started successfully!")

# PySpark is the Python library for Apache Spark, which allows you to write Spark programs using Python.