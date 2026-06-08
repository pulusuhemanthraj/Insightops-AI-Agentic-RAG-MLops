from fastapi import APIRouter
import pandas as pd
from src.utils.config import PROCESSED_DIR, RAW_DIR

router = APIRouter()


@router.get("/metrics/kpis")
def get_kpis():
    path = PROCESSED_DIR / "daily_kpis.csv"
    if not path.exists():
        return {"message": "Run feature_engineering.py first"}
    kpis = pd.read_csv(path)
    return {
        "total_events": int(kpis["daily_events"].sum()),
        "avg_latency_ms": round(float(kpis["avg_latency_ms"].mean()), 2),
        "avg_conversion_rate": round(float(kpis["conversion_rate"].mean()), 4),
        "avg_failure_rate": round(float(kpis["failure_rate"].mean()), 4),
    }


@router.get("/tickets/summary")
def ticket_summary():
    tickets = pd.read_csv(RAW_DIR / "support_tickets.csv")
    return {
        "total_tickets": int(len(tickets)),
        "by_severity": tickets["severity"].value_counts().to_dict(),
        "by_issue_type": tickets["issue_type"].value_counts().head(10).to_dict(),
    }
