"""FastAPI service exposing the multi-agent Research Assistant.

`POST /ask {question}` -> `{answer, route, review_verdict, metrics}`. It wraps the
same `root_agent` the CLI runs. Deploy target: Docker -> Cloud Run, where Vertex /
Gemini auth comes from the service's GCP identity (ADC), not a checked-in key.

Run locally:  uvicorn src.main:app --reload --port 8080

Cost guards on /ask (each call spends Gemini + Tavily credit):
  - API_KEY: when set, callers must send a matching `X-API-Key` header (the
    Streamlit UI sends it). Unset = open, for local dev only.
  - Question length capped at MAX_QUESTION_CHARS.
  - RATE_LIMIT_PER_MIN: max /ask calls per minute per instance (all callers).
"""
import os
import secrets
import time
from collections import deque

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import run_query

API_KEY = os.getenv("API_KEY", "")
MAX_QUESTION_CHARS = 1000
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
_recent_calls: deque[float] = deque()


def guard(x_api_key: str | None = Header(default=None)) -> None:
    """Reject callers without the shared key, then apply the rate limit."""
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    now = time.monotonic()
    while _recent_calls and now - _recent_calls[0] > 60:
        _recent_calls.popleft()
    if len(_recent_calls) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded, try again shortly")
    _recent_calls.append(now)

app = FastAPI(
    title="Multi-Agent Research Assistant",
    description=(
        "Routes research questions across a FAISS corpus of AI/ML papers (RAG) and "
        "live web search (Tavily), with a groundedness reviewer. Built on Google ADK."
    ),
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class AskResponse(BaseModel):
    question: str
    answer: str | None
    route: str | None
    review_verdict: str | None
    metrics: dict


@app.get("/health")
def health() -> dict:
    """Liveness probe (no model calls) — used by Cloud Run / load checks."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(guard)])
async def ask(req: AskRequest) -> dict:
    """Answer a research question end-to-end through the agent pipeline."""
    return await run_query(req.question)
