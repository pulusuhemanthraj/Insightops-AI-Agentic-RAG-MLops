from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.rag.document_loader import load_documents
from src.utils.config import VECTOR_STORE_PATH


class LocalVectorStore:
    """Lightweight local vector store using TF-IDF for offline portfolio demos.

    The project also includes requirements for FAISS/Chroma/LangChain/LlamaIndex so this module can be upgraded
    to embedding-based retrieval when API keys or local embedding models are available.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.docs = []
        self.matrix = None

    def build(self) -> None:
        self.docs = load_documents()
        texts = [d.text for d in self.docs]
        self.matrix = self.vectorizer.fit_transform(texts)

    def save(self) -> None:
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        with open(VECTOR_STORE_PATH / "tfidf_store.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "docs": self.docs, "matrix": self.matrix}, f)

    def load(self) -> None:
        with open(VECTOR_STORE_PATH / "tfidf_store.pkl", "rb") as f:
            obj = pickle.load(f)
        self.vectorizer = obj["vectorizer"]
        self.docs = obj["docs"]
        self.matrix = obj["matrix"]

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.matrix is None:
            self.load()
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idxs = np.argsort(scores)[::-1][:k]
        return [
            {"doc_id": self.docs[i].doc_id, "source": self.docs[i].source, "text": self.docs[i].text, "score": float(scores[i])}
            for i in idxs
        ]


def build_vector_store() -> None:
    store = LocalVectorStore()
    store.build()
    store.save()


if __name__ == "__main__":
    build_vector_store()
    print(f"Vector store created at {VECTOR_STORE_PATH}")
