from __future__ import annotations

from src.rag.retriever import retrieve


def answer_question(question: str) -> dict:
    contexts = retrieve(question, k=4)
    if not contexts:
        return {"answer": "No relevant context found.", "citations": []}

    context_text = "\n\n".join([f"[{c['source']}] {c['text']}" for c in contexts])
    answer = (
        "Based on the retrieved project knowledge base, "
        "the most relevant explanation is: "
        f"{contexts[0]['text'][:450]}"
    )
    if "checkout" in question.lower() or "payment" in question.lower():
        answer += " Recommended triage: inspect checkout-service and payment-service error logs, payment_failure events, latency spikes, and recent deployments."
    elif "rag" in question.lower() or "hallucination" in question.lower():
        answer += " Recommended evaluation: track faithfulness, relevance, context precision/recall, latency, and citation coverage."

    return {
        "question": question,
        "answer": answer,
        "citations": [{"source": c["source"], "score": c["score"], "snippet": c["text"][:220]} for c in contexts],
        "context_used": context_text[:1500],
    }
