from src.rag.vector_store import LocalVectorStore, build_vector_store
from src.utils.config import VECTOR_STORE_PATH


def retrieve(query: str, k: int = 4) -> list[dict]:
    if not (VECTOR_STORE_PATH / "tfidf_store.pkl").exists():
        build_vector_store()
    store = LocalVectorStore()
    return store.search(query, k=k)
