"""
Monroe Institute Archives — FastAPI Backend
=============================================
REST API for the consciousness exploration chat portal.
Lovable (or any frontend) calls these endpoints.
"""

import os
import json
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import anthropic
import voyageai
from pinecone import Pinecone
from bm25_index import BM25Index

# ── Config ──────────────────────────────────────────────────────────────────

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "monroe-explorer")
VOYAGE_MODEL = "voyage-3"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TOP_K = 8

SYSTEM_PROMPT = """You are a knowledgeable guide to the Monroe Institute archives — a collection of 770+ transcripts from consciousness research spanning 1974 to present, including Explorer Sessions, Robert Monroe's talks, CIA Gateway studies, and military remote viewing sessions.

YOUR PRIORITY: Tell a clear, coherent story. You are NOT a search engine — you are a storyteller and researcher. The user wants to understand, not just see quotes.

APPROACH:
1. Lead with a clear, direct answer to the question in your own words — like explaining it to a curious friend.
2. Weave in 2-3 of the strongest supporting quotes naturally within your narrative. Don't dump all sources — pick the most compelling ones.
3. Cite sources parenthetically — e.g., (Robert Monroe, Gateway Voyage talk) or (Explorer SHE, Session 6) — but keep citations light. The story matters more than the bibliography.
4. Build connective tissue between ideas. If multiple explorers describe similar things, say so: "This wasn't isolated — several explorers independently reported..." rather than listing each one separately.
5. End with an insight, implication, or invitation to explore further — not just a summary of what you quoted.

TONE: Warm, knowledgeable, slightly awed — like a seasoned researcher sharing genuinely fascinating findings over coffee. Not academic, not breathless.

WHEN CONTEXT IS THIN: If the transcript excerpts don't contain much on the topic, be honest: "The archives touch on this lightly, but here's what we find..." Then suggest 2-3 related topics the archives cover deeply.

WHAT NOT TO DO:
- Don't list every source with a quote — pick the best 2-3
- Don't organize by source (Explorer A said X, Explorer B said Y) — organize by idea
- Don't start with "Based on the transcript excerpts provided" — just answer naturally
- Don't repeat the same point from different sources — synthesize into one clear statement"""

# ── Exploration Paths ────────────────────────────────────────────────────────

EXPLORATION_PATHS = {
    "hero": {"icon": "👁", "label": "What did people actually experience outside their bodies?"},
    "sections": {
        "Start Here": [
            {"icon": "🌀", "label": "How did these experiments actually change people's consciousness?"},
            {"icon": "🧭", "label": "What happens as people go deeper into these states?"},
            {"icon": "📂", "label": "What's inside these archived sessions?"},
            {"icon": "🧠", "label": "What were researchers trying to figure out?"},
        ],
        "Most Surprising": [
            {"icon": "👁", "label": "What did people actually see outside their bodies?"},
            {"icon": "💀", "label": "Did anyone report experiences after death?"},
            {"icon": "🚪", "label": "How far did these experiences go?"},
            {"icon": "🧠", "label": "What changed people after these sessions?"},
            {"icon": "🧩", "label": "What were the most unexpected discoveries?"},
        ],
        "Military Files": [
            {"icon": "⚡", "label": "What happened during the military's remote viewing sessions?"},
            {"icon": "📡", "label": "What did the CIA Gateway research actually find?"},
            {"icon": "🧪", "label": "What were these experiments trying to prove?"},
            {"icon": "📁", "label": "What did the military conclude from all of this?"},
        ],
        "Strange & Unexplained": [
            {"icon": "👻", "label": "Strangest experiences in the archives"},
            {"icon": "🔮", "label": "Did anyone report contact with non-physical entities?"},
            {"icon": "🌀", "label": "Experiences that couldn't be explained"},
            {"icon": "🧠", "label": "Patterns across hundreds of sessions"},
        ],
        "Go Deeper": [
            {"icon": "🧭", "label": "What are Focus levels and how do they work?"},
            {"icon": "🔮", "label": "Who — or what — is Miranon?"},
            {"icon": "📖", "label": "Full session breakdowns"},
            {"icon": "🧬", "label": "Recurring themes across transcripts"},
        ],
    },
}

# ── Global clients (initialized on startup) ─────────────────────────────────

claude_client = None
voyage_client = None
pinecone_index = None
bm25_index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize API clients on startup."""
    global claude_client, voyage_client, pinecone_index, bm25_index

    print("Initializing API clients...")
    claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    bm25_index = BM25Index("bm25_corpus.json")
    print("All clients initialized.")
    yield
    print("Shutting down.")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Monroe Archives API",
    description="RAG API for the Monroe Institute consciousness exploration archives",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lovable dev + production domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    stream: bool = True


class Source(BaseModel):
    text: str
    explorer_id: str
    session_number: str
    filename: str
    topics: str
    score: float
    label: str


class ConfidenceInfo(BaseModel):
    level: str  # "high", "moderate", "low"
    avg_score: float


class ChatResponse(BaseModel):
    """Non-streaming response."""
    response: str
    sources: list[Source]
    confidence: ConfidenceInfo
    search_query: str  # The rewritten query used for search


# ── Core RAG Logic ───────────────────────────────────────────────────────────

def format_source_label(ctx: dict) -> str:
    """Create a readable source label from context metadata."""
    explorer_id = ctx.get("explorer_id", "Unknown")
    filename = ctx.get("filename", "")
    session_number = ctx.get("session_number", 0)

    if explorer_id in ("Professional_Seminars", "Quarterly_Tapes"):
        name = str(session_number) if session_number else filename
        for suffix in [".txt", ".pdf"]:
            name = name.replace(suffix, "")
        category = "Professional Seminar" if explorer_id == "Professional_Seminars" else "Quarterly Tape"
        return f'{category}: "{name}"'

    if explorer_id.startswith("RAM"):
        name = filename
        for suffix in [".pdf", "_djvu.txt", " (Transcript)", "(Transcript)"]:
            name = name.replace(suffix, "")
        name = name.strip()
        if explorer_id == "RAM_Gateway":
            return f'Robert Monroe, Gateway Voyage: "{name}"'
        elif explorer_id == "RAM_Guidelines":
            return f'Robert Monroe, Guidelines: "{name}"'
        elif explorer_id == "RAM_INSCOM":
            return f'Robert Monroe, INSCOM: "{name}"'
        else:
            return f'Robert Monroe: "{name}"'

    if session_number and session_number != 0:
        return f"Explorer {explorer_id}, Session {session_number}"
    else:
        name = filename.replace(".pdf", "").replace(".txt", "").strip()
        return f'Explorer {explorer_id}: "{name}"'


def rewrite_query_with_context(query: str, history: list[ChatMessage]) -> str:
    """Rewrite a follow-up question into a standalone query using conversation history."""
    if not history:
        return query

    recent = history[-4:]
    history_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:300]}"
        for m in recent
    )

    rewrite_prompt = f"""Given this conversation history and a follow-up question, rewrite the follow-up into a standalone search query that captures the full intent. Keep it concise (1-2 sentences max). Only return the rewritten query, nothing else.

Conversation history:
{history_text}

Follow-up question: {query}

Standalone search query:"""

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": rewrite_prompt}],
    )
    return response.content[0].text.strip()


def retrieve_context(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Hybrid retrieval: vector search + BM25 keyword search + Voyage reranking."""

    # Stage 1: Vector search (top 25)
    result = voyage_client.embed([query], model=VOYAGE_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    vector_results = pinecone_index.query(
        vector=query_embedding,
        top_k=25,
        include_metadata=True,
    )

    all_candidates = {}
    for match in vector_results.matches:
        meta = match.metadata
        all_candidates[match.id] = {
            "id": match.id,
            "text": meta.get("text", ""),
            "explorer_id": meta.get("explorer_id", "Unknown"),
            "session_number": meta.get("session_number", "?"),
            "filename": meta.get("filename", ""),
            "topics": meta.get("topics", ""),
            "vector_score": match.score,
            "vector_rank": None,
            "bm25_rank": None,
        }

    for rank, match in enumerate(vector_results.matches):
        all_candidates[match.id]["vector_rank"] = rank

    # Stage 2: BM25 keyword search (top 25)
    if bm25_index and bm25_index.is_ready:
        bm25_results = bm25_index.search(query, top_k=25)
        for rank, (chunk_id, bm25_score) in enumerate(bm25_results):
            if chunk_id in all_candidates:
                all_candidates[chunk_id]["bm25_rank"] = rank
            else:
                try:
                    fetch_result = pinecone_index.fetch(ids=[chunk_id])
                    if chunk_id in fetch_result.vectors:
                        vec = fetch_result.vectors[chunk_id]
                        meta = vec.metadata
                        all_candidates[chunk_id] = {
                            "id": chunk_id,
                            "text": meta.get("text", ""),
                            "explorer_id": meta.get("explorer_id", "Unknown"),
                            "session_number": meta.get("session_number", "?"),
                            "filename": meta.get("filename", ""),
                            "topics": meta.get("topics", ""),
                            "vector_score": 0.0,
                            "vector_rank": None,
                            "bm25_rank": rank,
                        }
                except Exception:
                    pass

    # Stage 3: Reciprocal Rank Fusion
    K = 60
    for cid, candidate in all_candidates.items():
        rrf = 0.0
        if candidate["vector_rank"] is not None:
            rrf += 1.0 / (K + candidate["vector_rank"])
        if candidate["bm25_rank"] is not None:
            rrf += 1.0 / (K + candidate["bm25_rank"])
        candidate["rrf_score"] = rrf

    sorted_candidates = sorted(all_candidates.values(), key=lambda x: x["rrf_score"], reverse=True)
    top_candidates = sorted_candidates[:20]

    # Stage 4: Voyage reranking
    try:
        docs = [c["text"] for c in top_candidates]
        if docs:
            rerank_result = voyage_client.rerank(
                query=query,
                documents=docs,
                model="rerank-2",
                top_k=top_k,
            )
            reranked = []
            for r in rerank_result.results:
                candidate = top_candidates[r.index]
                reranked.append({
                    "text": candidate["text"],
                    "explorer_id": candidate["explorer_id"],
                    "session_number": candidate["session_number"],
                    "filename": candidate["filename"],
                    "topics": candidate["topics"],
                    "score": r.relevance_score,
                })
            return reranked
    except Exception:
        pass

    # Fallback: return by RRF score
    contexts = []
    for candidate in top_candidates[:top_k]:
        contexts.append({
            "text": candidate["text"],
            "explorer_id": candidate["explorer_id"],
            "session_number": candidate["session_number"],
            "filename": candidate["filename"],
            "topics": candidate["topics"],
            "score": candidate.get("vector_score", 0.0),
        })
    return contexts


def build_context_prompt(contexts: list[dict]) -> str:
    """Format retrieved contexts into a prompt section."""
    if not contexts:
        return "No relevant transcript excerpts were found for this query."

    parts = ["Here are the most relevant transcript excerpts:\n"]
    for i, ctx in enumerate(contexts, 1):
        source_label = format_source_label(ctx)
        parts.append(f"--- Excerpt {i} ({source_label}, "
                     f"Relevance: {ctx['score']:.2f}) ---")
        parts.append(ctx["text"])
        if ctx["topics"]:
            parts.append(f"[Topics: {ctx['topics']}]")
        parts.append("")

    return "\n".join(parts)


def assess_confidence(contexts: list[dict]) -> ConfidenceInfo:
    """Assess confidence based on relevance scores."""
    if not contexts:
        return ConfidenceInfo(level="low", avg_score=0.0)
    avg_score = sum(c["score"] for c in contexts) / len(contexts)
    strong_matches = sum(1 for c in contexts if c["score"] >= 0.60)
    if avg_score >= 0.60 or strong_matches >= 3:
        return ConfidenceInfo(level="high", avg_score=avg_score)
    elif avg_score >= 0.40:
        return ConfidenceInfo(level="moderate", avg_score=avg_score)
    else:
        return ConfidenceInfo(level="low", avg_score=avg_score)


def build_claude_messages(query: str, contexts: list[dict], history: list[ChatMessage]) -> list[dict]:
    """Build the messages list for Claude."""
    context_prompt = build_context_prompt(contexts)

    user_message = f"""Based on the following transcript excerpts from the Monroe Institute archives, please answer the user's question.

{context_prompt}

User's question: {query}"""

    messages = []
    for msg in history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    return messages


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check — verifies all services are connected."""
    stats = pinecone_index.describe_index_stats() if pinecone_index else None
    return {
        "status": "ok",
        "services": {
            "claude": claude_client is not None,
            "voyage": voyage_client is not None,
            "pinecone": pinecone_index is not None,
            "bm25": bm25_index.is_ready if bm25_index else False,
        },
        "vectors": stats.total_vector_count if stats else 0,
    }


@app.get("/api/suggestions")
async def suggestions():
    """Return exploration path suggestions for the UI."""
    return EXPLORATION_PATHS


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint. Supports both streaming (SSE) and non-streaming."""
    if not claude_client:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Rewrite follow-up queries for better retrieval
    search_query = rewrite_query_with_context(request.query, request.history) \
        if request.history else request.query

    # Retrieve context
    contexts = retrieve_context(search_query, top_k=request.top_k)

    # Build sources for response
    sources = [
        Source(
            text=ctx["text"],
            explorer_id=ctx["explorer_id"],
            session_number=str(ctx["session_number"]),
            filename=ctx["filename"],
            topics=ctx["topics"],
            score=ctx["score"],
            label=format_source_label(ctx),
        )
        for ctx in contexts
    ]

    confidence = assess_confidence(contexts)

    if request.stream:
        # Server-Sent Events streaming
        async def event_stream():
            # First, send metadata (sources + confidence) as a JSON event
            meta = {
                "type": "metadata",
                "sources": [s.model_dump() for s in sources],
                "confidence": confidence.model_dump(),
                "search_query": search_query,
            }
            yield f"data: {json.dumps(meta)}\n\n"

            # Then stream the response text
            messages = build_claude_messages(request.query, contexts, request.history)
            with claude_client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    chunk = {"type": "text", "content": text}
                    yield f"data: {json.dumps(chunk)}\n\n"

            # Signal end
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming: return full response at once
        messages = build_claude_messages(request.query, contexts, request.history)
        response = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        return ChatResponse(
            response=response.content[0].text,
            sources=sources,
            confidence=confidence,
            search_query=search_query,
        )


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
