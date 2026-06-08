from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from src.utils.config import RAW_DIR, PROCESSED_DIR


def detect_log_anomalies() -> pd.DataFrame:
    logs = pd.read_csv(RAW_DIR / "application_logs.csv")
    logs["level_score"] = logs["level"].map({"INFO": 0, "WARN": 1, "ERROR": 2})
    features = logs[["latency_ms", "level_score"]]
    model = IsolationForest(contamination=0.08, random_state=42)
    logs["anomaly"] = model.fit_predict(features)
    anomalies = logs[logs["anomaly"] == -1].copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    anomalies.to_csv(PROCESSED_DIR / "log_anomalies.csv", index=False)
    return anomalies


if __name__ == "__main__":
    print(detect_log_anomalies().head())
