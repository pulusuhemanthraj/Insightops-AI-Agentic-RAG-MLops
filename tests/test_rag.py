from src.rag.rag_pipeline import answer_question


def test_rag_answer_shape():
    result = answer_question("How should RAG evaluation work?")
    assert "answer" in result
    assert "citations" in result
