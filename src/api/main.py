from fastapi import FastAPI
from src.api.routes_rag import router as rag_router
from src.api.routes_analytics import router as analytics_router
from src.api.routes_ml import router as ml_router

app = FastAPI(
    title="InsightOps AI API",
    description="Agentic RAG, product analytics, ML prediction, and MLOps API",
    version="1.0.0",
)

app.include_router(rag_router)
app.include_router(analytics_router)
app.include_router(ml_router)


@app.get("/health")
def health():
    return {"status": "healthy", "service": "InsightOps AI"}
