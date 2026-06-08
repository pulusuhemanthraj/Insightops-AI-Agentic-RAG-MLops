# Setup Guide

## Windows Command Prompt

```cmd
cd /d "%USERPROFILE%\Downloads"
git clone https://github.com/YOUR_USERNAME/insightops-ai-agentic-rag-mlops.git
cd insightops-ai-agentic-rag-mlops
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_generation.generate_synthetic_data
python -m src.pipelines.feature_engineering
python -m src.ml.train_ticket_severity_model
python -m src.ml.train_churn_model
python -m src.ml.anomaly_detection
python -m src.rag.document_loader
python -m src.rag.vector_store
python -m src.rag.evaluate_rag
uvicorn src.api.main:app --reload
```

Open another Command Prompt:

```cmd
cd /d "%USERPROFILE%\Downloads\insightops-ai-agentic-rag-mlops"
.venv\Scripts\activate
streamlit run dashboard\app.py
```
