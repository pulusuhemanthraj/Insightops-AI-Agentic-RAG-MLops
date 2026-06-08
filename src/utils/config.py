from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KB_DIR = DATA_DIR / "knowledge_base"
MODEL_DIR = ROOT_DIR / "models"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROCESSED_DIR / 'insightops.db'}")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", str(PROCESSED_DIR / "vector_store")))

for path in [RAW_DIR, PROCESSED_DIR, KB_DIR, MODEL_DIR, VECTOR_STORE_PATH]:
    path.mkdir(parents=True, exist_ok=True)
