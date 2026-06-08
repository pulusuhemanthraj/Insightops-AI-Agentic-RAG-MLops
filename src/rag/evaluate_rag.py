from __future__ import annotations

import time
import pandas as pd
from src.rag.rag_pipeline import answer_question
from src.utils.config import PROCESSED_DIR

EVAL_QUESTIONS = [
    "Why did checkout errors increase?",
    "How should RAG faithfulness be evaluated?",
    "What are the main product analytics KPIs?",
    "How should ML model governance be handled?",
]


def evaluate() -> pd.DataFrame:
    rows = []
    for q in EVAL_QUESTIONS:
        start = time.time()
        ans = answer_question(q)
        latency = time.time() - start
        rows.append({
            "question": q,
            "latency_seconds": round(latency, 4),
            "citation_count": len(ans.get("citations", [])),
            "has_answer": bool(ans.get("answer")),
            "hallucination_risk": "low" if len(ans.get("citations", [])) >= 2 else "medium",
        })
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_DIR / "rag_evaluation_results.csv", index=False)
    return df


if __name__ == "__main__":
    print(evaluate())
