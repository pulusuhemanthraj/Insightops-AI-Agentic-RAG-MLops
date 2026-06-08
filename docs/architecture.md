# Architecture

InsightOps AI follows a layered design:

1. **Data Layer**: Synthetic product events, support tickets, application logs, A/B test records, and markdown knowledge-base documents.
2. **Pipeline Layer**: Data validation, KPI tables, ticket features, log features, Spark-ready ETL, and Airflow orchestration.
3. **ML Layer**: Ticket severity classification, churn prediction, and log anomaly detection. Experiments are tracked with MLflow.
4. **RAG Layer**: Document chunking, local TF-IDF retrieval for offline use, FAISS/Chroma/LangChain/LlamaIndex-ready structure, citation-based answering, and RAG evaluation.
5. **API Layer**: FastAPI exposes RAG, analytics, ML prediction, ticket summary, and anomaly endpoints.
6. **Dashboard Layer**: Streamlit gives an executive KPI, product analytics, ticket, log, RAG, and evaluation UI.
