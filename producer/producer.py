# kafka-producer/producer.py

from kafka import KafkaProducer
import pandas as pd
import json
import time

# Configure Kafka producer
producer = KafkaProducer(
    bootstrap_servers='kafka:9092',   # kafka is the service name from docker-compose.yml (we'll setup later)
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load transaction data
df = pd.read_csv('creditcard.csv')

# Drop label (Class) because we are simulating real-world incoming transactions (without known fraud info)
df = df.drop('Class', axis=1)

# Stream transactions one by one
for idx, row in df.iterrows():
    message = row.to_dict()
    producer.send('fraud_topic', value=message)
    print(f"Sent transaction {idx}")
    time.sleep(0.05)  # 50ms delay to simulate streaming

producer.flush()
producer.close()
