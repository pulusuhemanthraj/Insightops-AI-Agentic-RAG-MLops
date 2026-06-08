"""Airflow DAG for InsightOps AI.

Place this folder in your Airflow DAGs path. The DAG orchestrates data generation,
feature engineering, ML training, anomaly detection, vector-store build, and RAG evaluation.
"""
from __future__ import annotations

from datetime import datetime
try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except Exception:  # Allows repo import without Airflow installed.
    DAG = None
    BashOperator = None

if DAG:
    with DAG(
        dag_id="insightops_ai_pipeline",
        start_date=datetime(2026, 1, 1),
        schedule="@daily",
        catchup=False,
        tags=["data-science", "rag", "mlops"],
    ) as dag:
        generate_data = BashOperator(task_id="generate_data", bash_command="python -m src.data_generation.generate_synthetic_data")
        features = BashOperator(task_id="feature_engineering", bash_command="python -m src.pipelines.feature_engineering")
        severity = BashOperator(task_id="train_ticket_severity", bash_command="python -m src.ml.train_ticket_severity_model")
        churn = BashOperator(task_id="train_churn", bash_command="python -m src.ml.train_churn_model")
        anomalies = BashOperator(task_id="detect_anomalies", bash_command="python -m src.ml.anomaly_detection")
        vector_store = BashOperator(task_id="build_vector_store", bash_command="python -m src.rag.vector_store")
        rag_eval = BashOperator(task_id="evaluate_rag", bash_command="python -m src.rag.evaluate_rag")

        generate_data >> features >> [severity, churn, anomalies] >> vector_store >> rag_eval
