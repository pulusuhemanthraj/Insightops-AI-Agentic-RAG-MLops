from __future__ import annotations

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from src.utils.config import RAW_DIR, MODEL_DIR, MLFLOW_TRACKING_URI


def build_user_features() -> pd.DataFrame:
    events = pd.read_csv(RAW_DIR / "product_events.csv")
    events["is_success"] = (events["event_name"] == "payment_success").astype(int)
    events["is_failure"] = (events["event_name"] == "payment_failure").astype(int)
    users = events.groupby("user_id").agg(
        event_count=("event_id", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        successes=("is_success", "sum"),
        failures=("is_failure", "sum"),
        sessions=("session_id", "nunique"),
    ).reset_index()
    risk = (users["failures"] * 0.25 + (users["avg_latency_ms"] > 450).astype(int) * 0.25 - users["successes"] * 0.03)
    threshold = np.quantile(risk, 0.70)
    users["churned"] = (risk > threshold).astype(int)
    return users


def train() -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("user-churn-prediction")
    df = build_user_features()
    features = ["event_count", "avg_latency_ms", "successes", "failures", "sessions"]
    X_train, X_test, y_train, y_test = train_test_split(df[features], df["churned"], test_size=0.2, random_state=42, stratify=df["churned"])
    model = GradientBoostingClassifier(random_state=42)
    with mlflow.start_run():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1", f1)
        path = MODEL_DIR / "churn_model.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))
    return {"roc_auc": auc, "f1": f1, "model_path": str(path)}


if __name__ == "__main__":
    print(train())
