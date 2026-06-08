@echo off
REM Windows Command Prompt helper for InsightOps AI
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data_generation.generate_synthetic_data
python -m src.pipelines.feature_engineering
python -m src.ml.train_ticket_severity_model
python -m src.ml.train_churn_model
python -m src.ml.anomaly_detection
python -m src.rag.vector_store
python -m src.rag.evaluate_rag
echo Setup completed. Start API: uvicorn src.api.main:app --reload
echo Start Dashboard in another CMD: streamlit run dashboard\app.py
