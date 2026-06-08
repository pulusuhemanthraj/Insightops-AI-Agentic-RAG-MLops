from pathlib import Path
from src.utils.config import MODEL_DIR


def list_models() -> list[str]:
    return [p.name for p in Path(MODEL_DIR).glob("*.joblib")]
