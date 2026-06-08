from __future__ import annotations

import pandas as pd
from src.rag.rag_pipeline import answer_question
from src.utils.config import PROCESSED_DIR, RAW_DIR


class RouterAgent:
    def route(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ["kpi", "metric", "conversion", "funnel", "users"]):
            return "analytics"
        if any(w in q for w in ["log", "error", "incident", "latency", "timeout"]):
            return "logs"
        return "rag"


class SQLAnalyticsAgent:
    def run(self) -> dict:
        kpis = pd.read_csv(PROCESSED_DIR / "daily_kpis.csv")
        return {
            "daily_events": int(kpis["daily_events"].sum()),
            "avg_latency_ms": round(float(kpis["avg_latency_ms"].mean()), 2),
            "avg_conversion_rate": round(float(kpis["conversion_rate"].mean()), 4),
            "top_product_area": kpis.groupby("product_area")["daily_events"].sum().idxmax(),
        }


class LogAnalysisAgent:
    def run(self) -> dict:
        logs = pd.read_csv(RAW_DIR / "application_logs.csv")
        errors = logs[logs["level"] == "ERROR"]
        return {
            "total_logs": int(len(logs)),
            "error_count": int(len(errors)),
            "top_error_service": errors["service"].value_counts().idxmax() if not errors.empty else None,
            "avg_error_latency_ms": round(float(errors["latency_ms"].mean()), 2) if not errors.empty else 0,
        }


class EvaluatorAgent:
    def evaluate(self, answer: dict) -> dict:
        citations = answer.get("citations", [])
        return {
            "citation_count": len(citations),
            "has_context": len(citations) > 0,
            "hallucination_risk": "low" if len(citations) >= 2 else "medium",
        }


def run_agentic_workflow(question: str) -> dict:
    route = RouterAgent().route(question)
    if route == "analytics":
        result = {"route": route, "answer": SQLAnalyticsAgent().run()}
    elif route == "logs":
        result = {"route": route, "answer": LogAnalysisAgent().run()}
    else:
        rag_answer = answer_question(question)
        result = {"route": route, "answer": rag_answer, "evaluation": EvaluatorAgent().evaluate(rag_answer)}
    return result
