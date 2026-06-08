from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd
from src.utils.config import MODEL_DIR, PROCESSED_DIR

router = APIRouter()


class SeverityRequest(BaseModel):
    priority_score: int
    resolution_hours: float
    text_length: int
    issue_type_encoded: int
    customer_segment_encoded: int
    status_encoded: int


@router.post("/predict/severity")
def predict_severity(req: SeverityRequest):
    model_path = MODEL_DIR / "ticket_severity_model.joblib"
    if not model_path.exists():
        return {"message": "Train model first: python src/ml/train_ticket_severity_model.py"}
    model = joblib.load(model_path)
    X = pd.DataFrame([req.model_dump()])
    return {"predicted_severity": model.predict(X)[0]}


@router.get("/logs/anomalies")
def log_anomalies():
    path = PROCESSED_DIR / "log_anomalies.csv"
    if not path.exists():
        return {"message": "Run anomaly_detection.py first"}
    df = pd.read_csv(path).head(25)
    return df.to_dict(orient="records")
