"""
main.py
-------
FastAPI application exposing the Agentic RAG support agent over HTTP.

Run:
    uvicorn main:app --reload --port 8000

Key endpoint:
    POST /api/chat/stream
        body: { "message": str, "chat_history": [{"role": "user"|"assistant", "content": str}] }
        returns: text/event-stream of the streamed answer, followed by a final
                 JSON metadata event (ticket_status, citations, escalation_reason)
"""

import os
import json
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from embedder import get_embedding_function, CHROMA_PERSIST_DIR, DEFAULT_COLLECTION
from agent_workflow import (
    init_state,
    transform_query,
    retrieve_documents,
    grade_documents,
    route_after_grading,
    stream_generate_answer,
    escalate,
    build_agent_graph,
)

load_dotenv()

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")

app = FastAPI(title="Autonomous Customer Support Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# ChromaDB connection (single shared client/collection for the app lifetime)
# ---------------------------------------------------------------------------

_chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)
_embedding_fn = get_embedding_function()
_collection = _chroma_client.get_or_create_collection(
    name=DEFAULT_COLLECTION,
    embedding_function=_embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

# Non-streaming compiled graph (used by /api/chat for simple request/response clients)
_compiled_graph = build_agent_graph(_collection)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[ChatTurn]] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    ticket_status: str
    escalation_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "kb_vector_count": _collection.count()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming endpoint: runs the full compiled LangGraph and returns the final state."""
    history = [turn.model_dump() for turn in (req.chat_history or [])]
    state = init_state(req.message, history)
    final_state = _compiled_graph.invoke(state)

    return ChatResponse(
        answer=final_state["generation"],
        citations=final_state["citations"],
        ticket_status=final_state["ticket_status"],
        escalation_reason=final_state.get("escalation_reason"),
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming endpoint. Runs transform -> retrieve -> grade synchronously (fast,
    non-generative steps), then either streams the generation token-by-token or
    yields a single escalation message — followed by a final metadata event.
    """
    history = [turn.model_dump() for turn in (req.chat_history or [])]
    state = init_state(req.message, history)

    async def event_stream():
        # 1) Query transformation
        nonlocal_state = transform_query(state)

        # 2) Retrieval
        nonlocal_state = retrieve_documents(nonlocal_state, _collection)

        # 3) Grading (self-correcting checkpoint)
        nonlocal_state = grade_documents(nonlocal_state)

        # 4) Conditional routing
        route = route_after_grading(nonlocal_state)

        if route == "generate_answer":
            async for delta in stream_generate_answer(nonlocal_state):
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

            final_meta = {
                "type": "done",
                "citations": nonlocal_state["citations"],
                "ticket_status": nonlocal_state["ticket_status"],
                "escalation_reason": None,
            }
            yield f"data: {json.dumps(final_meta)}\n\n"
        else:
            escalated_state = escalate(nonlocal_state)
            yield f"data: {json.dumps({'type': 'token', 'content': escalated_state['generation']})}\n\n"
            final_meta = {
                "type": "done",
                "citations": [],
                "ticket_status": escalated_state["ticket_status"],
                "escalation_reason": escalated_state["escalation_reason"],
            }
            yield f"data: {json.dumps(final_meta)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
