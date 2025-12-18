from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("Bigdata") \
    .master("local[*]") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Test the Spark session
print("Spark session started successfully!")


