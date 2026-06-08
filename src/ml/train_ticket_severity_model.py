from __future__ import annotations

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from src.utils.config import PROCESSED_DIR, MODEL_DIR, MLFLOW_TRACKING_URI
from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURES = ["priority_score", "resolution_hours", "text_length", "issue_type_encoded", "customer_segment_encoded", "status_encoded"]


def train() -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("ticket-severity-classification")
    df = pd.read_csv(PROCESSED_DIR / "ticket_features.csv")
    X = df[FEATURES]
    y = df["severity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("weighted_f1", f1)
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / "ticket_severity_model.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))
    logger.info("Ticket severity model saved: %s", path)
    logger.info("Classification report:\n%s", classification_report(y_test, preds))
    return {"accuracy": acc, "weighted_f1": f1, "model_path": str(path)}


if __name__ == "__main__":
    train()
