from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from src.utils.config import KB_DIR, PROCESSED_DIR


@dataclass
class DocumentChunk:
    doc_id: str
    source: str
    text: str


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def load_documents() -> list[DocumentChunk]:
    docs: list[DocumentChunk] = []
    for path in Path(KB_DIR).glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(text)):
            docs.append(DocumentChunk(doc_id=f"{path.stem}_{idx}", source=path.name, text=chunk))
    return docs


def export_chunks() -> None:
    import pandas as pd
    docs = load_documents()
    df = pd.DataFrame([d.__dict__ for d in docs])
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(PROCESSED_DIR / "document_chunks.csv", index=False)


if __name__ == "__main__":
    export_chunks()
    print(f"Exported chunks to {PROCESSED_DIR / 'document_chunks.csv'}")
