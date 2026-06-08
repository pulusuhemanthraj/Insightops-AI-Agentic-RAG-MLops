from src.data_generation.generate_synthetic_data import main as generate_data
from src.pipelines.feature_engineering import main as feature_engineering
from src.ml.train_ticket_severity_model import train as train_severity
from src.ml.train_churn_model import train as train_churn
from src.ml.anomaly_detection import detect_log_anomalies
from src.rag.vector_store import build_vector_store
from src.rag.evaluate_rag import evaluate

if __name__ == "__main__":
    generate_data()
    feature_engineering()
    print(train_severity())
    print(train_churn())
    print(detect_log_anomalies().head())
    build_vector_store()
    print(evaluate())
