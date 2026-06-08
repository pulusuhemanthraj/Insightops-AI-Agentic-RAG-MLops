"""Kafka-style producer placeholder.

Use this to demonstrate streaming design. Install confluent-kafka and point it to a local broker when needed.
"""
import json
import time
import pandas as pd
from src.utils.config import RAW_DIR


def stream_events(delay_seconds: float = 0.05):
    events = pd.read_csv(RAW_DIR / "product_events.csv")
    for _, row in events.head(100).iterrows():
        payload = row.to_dict()
        print(json.dumps(payload))
        time.sleep(delay_seconds)


if __name__ == "__main__":
    stream_events()
