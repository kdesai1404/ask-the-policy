"""FastAPI service.  Run: uvicorn src.api:app --reload"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import pipeline

app = FastAPI(title="Ask-the-Policy", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    chunking: Literal["fixed", "semantic"] = "semantic"
    k: int = Field(5, ge=1, le=10)
    strict: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return pipeline.ask(req.question, req.mode, req.chunking, req.k, req.strict)
    except FileNotFoundError as e:  # index not built yet
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:  # e.g. missing API key
        raise HTTPException(status_code=500, detail=str(e))
