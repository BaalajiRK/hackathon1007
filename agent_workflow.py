"""
agent_workflow.py
------------------
The self-correcting Agentic RAG state machine, built with LangGraph.

Flow:
    START
      -> transform_query      (rewrite conversational message into a search query)
      -> retrieve_documents    (vector search against ChromaDB)
      -> grade_documents       (LLM grades each chunk: relevant / irrelevant)
      -> [conditional edge]
            if enough relevant chunks -> generate_answer -> END
            else                      -> escalate        -> END

Ticket states: "In Progress" -> "Resolved" | "Unresolved - Insufficient KB"
"""

import os
from enum import Enum
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

MIN_RELEVANT_CHUNKS = int(os.getenv("MIN_RELEVANT_CHUNKS", "1"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TicketStatus(str, Enum):
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    ESCALATED = "Unresolved - Insufficient KB"


class RetrievedChunk(BaseModel):
    content: str
    source: str
    page: int
    score: Optional[float] = None
    relevant: Optional[bool] = None


class GradeResult(BaseModel):
    """Structured output the grader LLM must produce for a single chunk."""
    relevant: bool = Field(description="True if this chunk contains information that helps answer the query")
    reasoning: str = Field(description="One short sentence explaining the grading decision")


class QueryRewrite(BaseModel):
    """Structured output for the query transformation node."""
    search_query: str = Field(description="A concise, keyword-rich query optimized for vector similarity search")


class AgentState(TypedDict):
    """The shared state object threaded through every node in the graph."""
    original_query: str
    chat_history: List[dict]          # [{"role": "user"|"assistant", "content": str}, ...]
    transformed_query: str
    retrieved_chunks: List[RetrievedChunk]
    relevant_chunks: List[RetrievedChunk]
    generation: str
    citations: List[str]
    ticket_status: str
    escalation_reason: Optional[str]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature, streaming=True)


# ---------------------------------------------------------------------------
# Node: Query Transformation
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite conversational customer support messages into concise, keyword-rich "
     "queries optimized for vector similarity search against a knowledge base. "
     "Resolve pronouns and vague references using the chat history. "
     "Do not answer the question — only produce the search query."),
    ("human",
     "Chat history:\n{chat_history}\n\n"
     "Latest user message: {query}\n\n"
     "Rewritten search query:"),
])


def transform_query(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0).with_structured_output(QueryRewrite)
    history_str = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in state.get("chat_history", [])
    ) or "(no prior turns)"

    chain = QUERY_REWRITE_PROMPT | llm
    result: QueryRewrite = chain.invoke({
        "chat_history": history_str,
        "query": state["original_query"],
    })

    state["transformed_query"] = result.search_query
    return state


# ---------------------------------------------------------------------------
# Node: Retrieval
# ---------------------------------------------------------------------------

def retrieve_documents(state: AgentState, collection) -> AgentState:
    """
    `collection` is a ChromaDB collection object injected at graph-build time
    (see build_agent_graph below) so this node stays testable/pure.
    """
    query = state["transformed_query"]
    results = collection.query(
        query_texts=[query],
        n_results=6,
        include=["documents", "metadatas", "distances"],
    )

    chunks: List[RetrievedChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for content, meta, dist in zip(docs, metas, dists):
        chunks.append(RetrievedChunk(
            content=content,
            source=meta.get("source", "unknown"),
            page=meta.get("page", 1),
            score=1 - dist,  # cosine distance -> similarity
        ))

    state["retrieved_chunks"] = chunks
    return state


# ---------------------------------------------------------------------------
# Node: Document Grading  (the self-correcting core)
# ---------------------------------------------------------------------------

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance grader for a customer support retrieval system. "
     "Given a user query and a single retrieved document chunk, decide if the chunk "
     "contains information that is directly useful and safe for answering the query. "
     "Be conservative: if the chunk is only tangentially related, mark it NOT relevant. "
     "This prevents the support agent from hallucinating answers off weak context."),
    ("human",
     "User query: {query}\n\nDocument chunk:\n{chunk}\n\nGrade this chunk."),
])


def grade_documents(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.0).with_structured_output(GradeResult)
    chain = GRADE_PROMPT | llm

    query = state["transformed_query"]
    relevant: List[RetrievedChunk] = []

    for chunk in state["retrieved_chunks"]:
        grade: GradeResult = chain.invoke({
            "query": query,
            "chunk": f"[Source: {chunk.source}, Page {chunk.page}]\n{chunk.content}",
        })
        chunk.relevant = grade.relevant
        if grade.relevant:
            relevant.append(chunk)

    state["relevant_chunks"] = relevant
    return state


def route_after_grading(state: AgentState) -> str:
    """Conditional edge: decide whether to generate an answer or escalate to a human."""
    if len(state["relevant_chunks"]) >= MIN_RELEVANT_CHUNKS:
        return "generate_answer"
    return "escalate"


# ---------------------------------------------------------------------------
# Node: Generation (with strict citation layer)
# ---------------------------------------------------------------------------

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a customer support agent. Answer the user's question using ONLY the "
     "provided context chunks below. Do not use outside knowledge.\n\n"
     "STRICT CITATION RULE: every factual claim must be followed by an inline citation "
     "in the exact format [Source: <filename>, Page <n>], using the source/page metadata "
     "given with each chunk. If different chunks support different sentences, cite each "
     "one separately. Never invent a source or page number that isn't in the context.\n\n"
     "Context:\n{context}"),
    ("human", "{query}"),
])


def _format_context(chunks: List[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Source: {c.source}, Page {c.page}]\n{c.content}" for c in chunks
    )


def generate_answer(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.1)
    chain = GENERATION_PROMPT | llm

    context = _format_context(state["relevant_chunks"])
    response = chain.invoke({
        "context": context,
        "query": state["original_query"],
    })

    state["generation"] = response.content
    state["citations"] = sorted({
        f"{c.source}, Page {c.page}" for c in state["relevant_chunks"]
    })
    state["ticket_status"] = TicketStatus.RESOLVED.value
    state["escalation_reason"] = None
    return state


async def stream_generate_answer(state: AgentState):
    """
    Async generator version used by the FastAPI streaming endpoint.
    Yields text deltas as they arrive from the LLM, then a final citations block.
    """
    llm = get_llm(temperature=0.1)
    chain = GENERATION_PROMPT | llm
    context = _format_context(state["relevant_chunks"])

    full_text = ""
    async for chunk in chain.astream({"context": context, "query": state["original_query"]}):
        delta = chunk.content or ""
        full_text += delta
        if delta:
            yield delta

    state["generation"] = full_text
    state["citations"] = sorted({
        f"{c.source}, Page {c.page}" for c in state["relevant_chunks"]
    })
    state["ticket_status"] = TicketStatus.RESOLVED.value


# ---------------------------------------------------------------------------
# Node: Escalation
# ---------------------------------------------------------------------------

def escalate(state: AgentState) -> AgentState:
    state["generation"] = (
        "I wasn't able to find reliable information in our knowledge base to answer "
        "this safely. I've flagged this conversation for a human support specialist "
        "to follow up with you shortly."
    )
    state["citations"] = []
    state["ticket_status"] = TicketStatus.ESCALATED.value
    state["escalation_reason"] = (
        f"No chunks met the relevance bar "
        f"({len(state['relevant_chunks'])}/{MIN_RELEVANT_CHUNKS} required) "
        f"for query: '{state['transformed_query']}'"
    )
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_agent_graph(collection):
    """
    Builds and compiles the LangGraph state machine.
    `collection` is a live ChromaDB collection, injected here (from main.py)
    so this module has no hard dependency on how the collection was created.
    """
    graph = StateGraph(AgentState)

    graph.add_node("transform_query", transform_query)
    graph.add_node("retrieve_documents", lambda state: retrieve_documents(state, collection))
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("escalate", escalate)

    graph.set_entry_point("transform_query")
    graph.add_edge("transform_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate_answer": "generate_answer", "escalate": "escalate"},
    )
    graph.add_edge("generate_answer", END)
    graph.add_edge("escalate", END)

    return graph.compile()


def init_state(original_query: str, chat_history: Optional[List[dict]] = None) -> AgentState:
    return AgentState(
        original_query=original_query,
        chat_history=chat_history or [],
        transformed_query="",
        retrieved_chunks=[],
        relevant_chunks=[],
        generation="",
        citations=[],
        ticket_status=TicketStatus.IN_PROGRESS.value,
        escalation_reason=None,
    )
