from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import random
import numpy as np
import pandas as pd
from src.utils.config import RAW_DIR, KB_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
random.seed(42)
np.random.seed(42)

PRODUCTS = ["search", "recommendations", "checkout", "payments", "profile", "notifications"]
SEGMENTS = ["free", "pro", "enterprise", "student"]
EVENTS = ["page_view", "click", "search", "add_to_cart", "checkout_start", "payment_success", "payment_failure"]
SEVERITIES = ["low", "medium", "high", "critical"]


def generate_product_events(n: int = 15000) -> pd.DataFrame:
    start = datetime.now() - timedelta(days=60)
    rows = []
    for i in range(n):
        ts = start + timedelta(minutes=random.randint(0, 60 * 24 * 60))
        product = random.choice(PRODUCTS)
        segment = random.choice(SEGMENTS)
        event = random.choices(EVENTS, weights=[35, 25, 15, 8, 8, 6, 3])[0]
        latency = max(20, np.random.normal(260, 80))
        if product == "checkout" and event in ["payment_failure", "checkout_start"]:
            latency += random.randint(80, 400)
        rows.append({
            "event_id": f"evt_{i:06d}",
            "user_id": f"user_{random.randint(1, 2500):05d}",
            "timestamp": ts.isoformat(),
            "product_area": product,
            "user_segment": segment,
            "event_name": event,
            "latency_ms": round(latency, 2),
            "session_id": f"sess_{random.randint(1, 5000):05d}",
            "device": random.choice(["web", "ios", "android"]),
            "country": random.choice(["US", "IN", "CA", "UK", "DE"]),
        })
    return pd.DataFrame(rows)


def generate_tickets(n: int = 2500) -> pd.DataFrame:
    issue_types = ["login", "checkout", "payment", "latency", "data_sync", "notification", "account"]
    rows = []
    for i in range(n):
        issue = random.choice(issue_types)
        severity = random.choices(SEVERITIES, weights=[45, 35, 15, 5])[0]
        if issue in ["payment", "checkout"]:
            severity = random.choices(SEVERITIES, weights=[20, 35, 30, 15])[0]
        resolution_hours = max(1, np.random.gamma(shape=2.0, scale=8.0))
        rows.append({
            "ticket_id": f"tkt_{i:05d}",
            "created_at": (datetime.now() - timedelta(days=random.randint(0, 60), hours=random.randint(0, 23))).isoformat(),
            "issue_type": issue,
            "severity": severity,
            "priority_score": random.randint(1, 100),
            "customer_segment": random.choice(SEGMENTS),
            "text": f"Customer reported {issue} issue with intermittent failures and degraded experience.",
            "resolution_hours": round(resolution_hours, 2),
            "status": random.choice(["open", "in_progress", "resolved", "closed"]),
        })
    return pd.DataFrame(rows)


def generate_logs(n: int = 6000) -> pd.DataFrame:
    services = ["auth-service", "payment-service", "checkout-service", "search-service", "recommendation-service"]
    levels = ["INFO", "WARN", "ERROR"]
    rows = []
    for i in range(n):
        service = random.choice(services)
        level = random.choices(levels, weights=[75, 17, 8])[0]
        if service in ["payment-service", "checkout-service"] and random.random() < 0.15:
            level = random.choice(["WARN", "ERROR"])
        msg = {
            "INFO": "request completed successfully",
            "WARN": "response latency above threshold",
            "ERROR": "dependency timeout or validation failure",
        }[level]
        rows.append({
            "log_id": f"log_{i:06d}",
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 60 * 24 * 30))).isoformat(),
            "service": service,
            "level": level,
            "message": msg,
            "latency_ms": round(max(10, np.random.normal(300 if level != "ERROR" else 900, 150)), 2),
            "request_id": f"req_{random.randint(1, 3000):05d}",
        })
    return pd.DataFrame(rows)


def generate_ab_tests(n: int = 2000) -> pd.DataFrame:
    rows = []
    for i in range(n):
        variant = random.choice(["control", "treatment"])
        base = 0.11 if variant == "control" else 0.135
        converted = np.random.binomial(1, base)
        rows.append({
            "experiment_id": "checkout_cta_experiment",
            "user_id": f"user_{i:05d}",
            "variant": variant,
            "converted": converted,
            "revenue": round(np.random.exponential(70) if converted else 0, 2),
        })
    return pd.DataFrame(rows)


def write_knowledge_base() -> None:
    docs = {
        "checkout_runbook.md": "Checkout Incident Runbook\nPayment failures usually increase when downstream payment APIs timeout, schema validation changes fail, or checkout latency crosses 800 ms. Triage by checking checkout-service and payment-service ERROR logs, payment_failure events, and recent deployments.",
        "rag_evaluation_guide.md": "RAG Evaluation Guide\nEvaluate answer faithfulness, answer relevance, context precision, context recall, latency, and citation coverage. Any answer that lacks retrieved context should be marked high hallucination risk.",
        "product_analytics_metrics.md": "Product Analytics Metrics\nCore KPIs include daily active users, conversion rate, checkout start rate, payment success rate, payment failure rate, average latency, cohort retention, funnel drop-off, and segment-level engagement.",
        "ml_model_governance.md": "ML Model Governance\nModels should be tracked with versioned parameters, metrics, artifacts, validation slices, feature importance, and production monitoring checks. Thresholds should be selected using precision and recall tradeoffs.",
    }
    for name, text in docs.items():
        (KB_DIR / name).write_text(text, encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    generate_product_events().to_csv(RAW_DIR / "product_events.csv", index=False)
    generate_tickets().to_csv(RAW_DIR / "support_tickets.csv", index=False)
    generate_logs().to_csv(RAW_DIR / "application_logs.csv", index=False)
    generate_ab_tests().to_csv(RAW_DIR / "ab_test_results.csv", index=False)
    write_knowledge_base()
    logger.info("Synthetic data and knowledge-base docs generated in %s", RAW_DIR)


if __name__ == "__main__":
    main()
