# spark-processor/spark_job.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, DoubleType
import requests
import sqlite3
import json


transaction_schema = StructType([
    StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)
] + [StructField("Amount", DoubleType(), True)])

spark = SparkSession.builder \
    .appName("FraudDetectionStreamProcessor") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2") \
    .getOrCreate()

db_file = "fraud_predictions.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS fraud_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        v1 REAL, v2 REAL, v3 REAL, v4 REAL, v5 REAL, v6 REAL, v7 REAL, v8 REAL, v9 REAL, v10 REAL, v11 REAL, v12 REAL, v13 REAL, v14 REAL, v15 REAL, v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL, v21 REAL, v22 REAL, v23 REAL, v24 REAL, v25 REAL, v26 REAL, v27 REAL, v28 REAL,
        amount REAL
    )
''')
conn.commit()
conn.close()


df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "fraud_topic") \
    .option("startingOffsets", "earliest") \
    .load()

transaction_df = df.selectExpr("CAST(value AS STRING) as json_value") \
    .select(from_json(col("json_value"), transaction_schema).alias("data")) \
    .select("data.*")

def process_row(row):
    try:
        data = {field: getattr(row, field) for field in transaction_schema.fieldNames()}
        response = requests.post("http://api:8000/predict", json=data)
        prediction = response.json()
        print(f"Transaction: {data} --> Fraud Prediction: {prediction}")
        if prediction['fraud']:
            conn = sqlite3.connect(db_file)
            placeholders = ', '.join(['?'] * len(data))
            query = f"INSERT INTO fraud_transactions ({', '.join(data.keys())}) VALUES ({placeholders})"
            conn.execute(query, list(data.values()))
            conn.commit()
            conn.close()
    except Exception as e:
        print(f"Error while calling API: {e}")

def foreach_batch_function(df, epoch_id):
    rows = df.collect()
    for row in rows:
        process_row(row)

query = transaction_df.writeStream \
    .foreachBatch(foreach_batch_function) \
    .start()

query.awaitTermination()
