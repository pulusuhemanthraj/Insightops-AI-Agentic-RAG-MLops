from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import plotly.express as px
from src.rag.agents import run_agentic_workflow
from src.utils.config import RAW_DIR, PROCESSED_DIR

st.set_page_config(page_title="InsightOps AI", page_icon="🤖", layout="wide")
st.title("InsightOps AI — Agentic RAG + Product Analytics + MLOps")
st.caption("Enterprise-style portfolio project for Data Science, GenAI/RAG, ML, FastAPI, Streamlit, Docker, Airflow, and MLflow.")

page = st.sidebar.radio("Navigate", ["Executive KPIs", "Product Analytics", "Tickets", "Logs", "RAG Assistant", "Model Evaluation"])


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.warning(f"Missing {path.name}. Run data generation and feature engineering first.")
        return pd.DataFrame()
    return pd.read_csv(path)

if page == "Executive KPIs":
    kpis = load_csv(PROCESSED_DIR / "daily_kpis.csv")
    if not kpis.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Events", f"{int(kpis.daily_events.sum()):,}")
        c2.metric("Avg Latency", f"{kpis.avg_latency_ms.mean():.1f} ms")
        c3.metric("Avg Conversion", f"{kpis.conversion_rate.mean():.2%}")
        c4.metric("Avg Failure", f"{kpis.failure_rate.mean():.2%}")
        fig = px.line(kpis, x="date", y="daily_events", color="product_area", title="Daily Events by Product Area")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Product Analytics":
    kpis = load_csv(PROCESSED_DIR / "daily_kpis.csv")
    if not kpis.empty:
        st.dataframe(kpis.tail(50), use_container_width=True)
        st.plotly_chart(px.bar(kpis.groupby("product_area", as_index=False)["daily_events"].sum(), x="product_area", y="daily_events", title="Events by Product Area"), use_container_width=True)

elif page == "Tickets":
    tickets = load_csv(RAW_DIR / "support_tickets.csv")
    if not tickets.empty:
        st.plotly_chart(px.histogram(tickets, x="severity", color="issue_type", title="Tickets by Severity and Issue Type"), use_container_width=True)
        st.dataframe(tickets.head(100), use_container_width=True)

elif page == "Logs":
    logs = load_csv(RAW_DIR / "application_logs.csv")
    if not logs.empty:
        st.plotly_chart(px.histogram(logs, x="service", color="level", title="Log Levels by Service"), use_container_width=True)
        st.dataframe(logs.head(100), use_container_width=True)

elif page == "RAG Assistant":
    question = st.text_input("Ask a business, log, analytics, or documentation question", "Why did checkout errors increase?")
    if st.button("Ask InsightOps AI"):
        result = run_agentic_workflow(question)
        st.json(result)

elif page == "Model Evaluation":
    rag_eval = load_csv(PROCESSED_DIR / "rag_evaluation_results.csv")
    if not rag_eval.empty:
        st.dataframe(rag_eval, use_container_width=True)
        st.plotly_chart(px.bar(rag_eval, x="question", y="latency_seconds", title="RAG Latency by Question"), use_container_width=True)
    else:
        st.info("Run: python src/rag/evaluate_rag.py")
