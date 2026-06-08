# InsightOps AI - Agentic RAG + Product Analytics + MLOps Platform

InsightOps AI is a complete end-to-end portfolio project for Data Science, Generative AI, RAG, Data Engineering, Product Analytics, and MLOps. It simulates an enterprise product analytics environment with product events, support tickets, application logs, knowledge-base documents, ML models, RAG pipelines, FastAPI services, Streamlit dashboards, MLflow experiment tracking, Docker, Airflow DAGs, and CI/CD.

## What this project demonstrates

- Product analytics: KPIs, funnels, cohorts, A/B testing, engagement metrics
- Data engineering: synthetic data generation, ETL, PySpark-ready transformations, Kafka-style ingestion, Airflow orchestration
- Machine learning: ticket severity classification, churn prediction, log anomaly detection
- GenAI/RAG: LangChain/LlamaIndex-ready document ingestion, FAISS/Chroma vector search, citation-based answers
- Agentic workflow: router, retriever, analytics, log-analysis, summarization, and evaluator agents
- Evaluation: ML metrics, RAG quality checks, latency, context precision, hallucination-risk checks
- MLOps: MLflow, Docker Compose, FastAPI, Streamlit, GitHub Actions, modular tests

## Architecture

```text
Synthetic Data --> ETL/Data Quality --> Feature Store Tables --> ML Models --> FastAPI
        |                 |                      |              |           |
        |                 |                      |              |           --> Streamlit Dashboard
        |                 |                      |              |
        |                 |                      |              --> MLflow Tracking
        |                 |
Knowledge Base Docs --> Chunking/Embeddings --> Vector Store --> Agentic RAG Assistant
```

## Quick Start Without Docker

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_generation.generate_synthetic_data
python -m src.pipelines.feature_engineering
python -m src.ml.train_ticket_severity_model
python -m src.ml.train_churn_model
python -m src.rag.document_loader
uvicorn src.api.main:app --reload
```

Open another terminal:

```bash
streamlit run dashboard/app.py
```

## Quick Start With Docker

```bash
docker compose up --build
```

Services:

- FastAPI: http://localhost:8000
- API docs: http://localhost:8000/docs
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000

## Example API Calls

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics/kpis
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"Why did checkout errors increase?"}'
```

## Recommended GitHub Repository Name

```text
insightops-ai-agentic-rag-mlops
```

## Resume Bullet

Built InsightOps AI, an end-to-end Agentic RAG and MLOps platform integrating product telemetry, support tickets, application logs, and knowledge-base documents to support KPI monitoring, incident analysis, ML prediction, and AI-powered document search using Python, SQL, Spark/PySpark-ready pipelines, LangChain/LlamaIndex-style RAG, FAISS/Chroma-ready vector search, FastAPI, Streamlit, Docker, MLflow, and Airflow.
