from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.config import RAW_DIR, PROCESSED_DIR
from src.pipelines.data_quality_checks import validate_not_empty, validate_required_columns, validate_null_rate
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_kpis(events: pd.DataFrame) -> pd.DataFrame:
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events["date"] = events["timestamp"].dt.date
    grouped = events.groupby(["date", "product_area"]).agg(
        daily_events=("event_id", "count"),
        daily_users=("user_id", "nunique"),
        avg_latency_ms=("latency_ms", "mean"),
        payment_failures=("event_name", lambda s: (s == "payment_failure").sum()),
        checkout_starts=("event_name", lambda s: (s == "checkout_start").sum()),
        payment_success=("event_name", lambda s: (s == "payment_success").sum()),
    ).reset_index()
    grouped["conversion_rate"] = grouped["payment_success"] / grouped["checkout_starts"].replace(0, pd.NA)
    grouped["failure_rate"] = grouped["payment_failures"] / grouped["daily_events"].replace(0, pd.NA)
    return grouped.fillna(0)


def build_ticket_features(tickets: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["text_length"] = df["text"].fillna("").str.len()
    for col in ["issue_type", "customer_segment", "status"]:
        df[f"{col}_encoded"] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def build_log_features(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.copy()
    df["is_error"] = (df["level"] == "ERROR").astype(int)
    df["is_warn"] = (df["level"] == "WARN").astype(int)
    service_error = df.groupby("service").agg(
        log_count=("log_id", "count"),
        error_count=("is_error", "sum"),
        warn_count=("is_warn", "sum"),
        avg_latency_ms=("latency_ms", "mean"),
    ).reset_index()
    service_error["error_rate"] = service_error["error_count"] / service_error["log_count"]
    return service_error


def main() -> None:
    events = pd.read_csv(RAW_DIR / "product_events.csv")
    tickets = pd.read_csv(RAW_DIR / "support_tickets.csv")
    logs = pd.read_csv(RAW_DIR / "application_logs.csv")

    validate_not_empty(events, "product_events")
    validate_required_columns(events, ["event_id", "timestamp", "event_name", "latency_ms"], "product_events")
    validate_null_rate(events)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    build_kpis(events).to_csv(PROCESSED_DIR / "daily_kpis.csv", index=False)
    build_ticket_features(tickets).to_csv(PROCESSED_DIR / "ticket_features.csv", index=False)
    build_log_features(logs).to_csv(PROCESSED_DIR / "service_log_features.csv", index=False)
    logger.info("Feature tables created in %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
