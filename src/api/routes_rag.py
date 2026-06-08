from fastapi import APIRouter
from pydantic import BaseModel
from src.rag.agents import run_agentic_workflow
from src.rag.rag_pipeline import answer_question

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    agentic: bool = True


@router.post("/ask")
def ask(req: AskRequest):
    if req.agentic:
        return run_agentic_workflow(req.question)
    return answer_question(req.question)
