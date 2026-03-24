"""
Monroe Institute Archives — Consciousness Exploration Portal
==============================================================
Streamlit-based RAG chat app powered by Claude + Voyage AI + Pinecone.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import anthropic
import voyageai
from pinecone import Pinecone

# ── Config ──────────────────────────────────────────────────────────────────

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "monroe-explorer")
VOYAGE_MODEL = "voyage-3"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TOP_K = 8

SYSTEM_PROMPT = """You are an expert research assistant specializing in the Monroe Institute's archives — including the Explorer Sessions (1974-1990), Robert Monroe's talks and interviews, Gateway Voyage and Guidelines program talks, Professional Seminars, and INSCOM military sessions.

You have deep knowledge of:
- Focus levels (Focus 10, 12, 15, 21, etc.) and their characteristics
- Hemi-Sync technology and its applications
- The various explorers (identified by codes like IMEC, SHE, GLA, AUB, etc.) and their unique experiences
- Channeled entities and communications (e.g., Miranon through SHE/Shay Ellsworth)
- Robert Monroe's personal experiences, philosophy, and teachings from his talks and interviews
- Gateway Voyage and Guidelines program content
- Professional Seminar presentations on consciousness research
- Topics including: astral travel, entity communication, healing, consciousness states, past lives, death/dying, spirit rescue, time/space perception, energy phenomena, and planetary consciousness

INSTRUCTIONS:
1. Ground your answers in the actual transcript excerpts provided as context. Always cite your sources using the source label provided with each excerpt and include a relevant quote.
2. When synthesizing across sessions, note which explorers/sessions contribute to each point.
3. If the context doesn't contain enough information to answer fully, say so honestly — don't fabricate details about the sessions.
4. Use a warm, knowledgeable tone — like a seasoned researcher sharing fascinating findings.
5. When asked about specific explorers, sessions, or phenomena, provide specific details from the transcripts.
6. If a question is about something not covered in the provided context, acknowledge that and offer to search for related topics.

FORMAT:
- Use clear paragraphs for readability
- Include direct quotes from transcripts when relevant, formatted with quotation marks
- Cite sources using the label from each excerpt — e.g., (SHE, Session 6) or (Robert Monroe, Gateway Voyage: "Saturday Night Talk")
- For cross-session themes, organize your response thematically"""

# ── Exploration Paths (grouped by depth) ────────────────────────────────────

HERO_CHIP = {"icon": "👁", "label": "What did people actually experience outside their bodies?"}

EXPLORATION_PATHS = {
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
}

# ── Custom CSS ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
.stApp {
    background: linear-gradient(165deg, #0a0a1a 0%, #0d1130 35%, #0a0a1a 70%, #110d24 100%);
    font-family: 'Inter', sans-serif;
}

/* Subtle animated gradient overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(88, 28, 135, 0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(30, 58, 138, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(88, 28, 135, 0.04) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
    animation: ambientShift 20s ease-in-out infinite alternate;
}

@keyframes ambientShift {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.8; }
}

/* ── Hide Streamlit defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d20 0%, #0a0a18 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.1) !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: rgba(200, 200, 230, 0.7) !important;
    font-size: 0.85rem !important;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: rgba(200, 200, 230, 0.9) !important;
    font-weight: 500 !important;
}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 4rem 2rem 2rem 2rem;
    max-width: 800px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}

.hero-glow {
    position: absolute;
    top: 20%;
    left: 50%;
    transform: translateX(-50%);
    width: 500px;
    height: 300px;
    background: radial-gradient(ellipse, rgba(139, 92, 246, 0.12) 0%, transparent 70%);
    pointer-events: none;
    filter: blur(40px);
}

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #e2e0f0;
    line-height: 1.2;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
    text-shadow: 0 0 60px rgba(139, 92, 246, 0.15);
}

.hero-subtitle {
    font-size: 1.1rem;
    color: rgba(180, 175, 210, 0.75);
    line-height: 1.6;
    max-width: 550px;
    margin: 0 auto 0.5rem auto;
    font-weight: 300;
}

.hero-credential {
    font-size: 0.8rem;
    color: rgba(139, 92, 246, 0.5);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 1.5rem;
    font-weight: 500;
}

.hero-classified {
    display: inline-block;
    margin-top: 1.2rem;
    padding: 0.5rem 1.2rem;
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: 8px;
    background: rgba(139, 92, 246, 0.06);
    backdrop-filter: blur(10px);
}

.hero-classified-text {
    font-size: 0.75rem;
    color: rgba(180, 175, 210, 0.6);
    letter-spacing: 0.05em;
    line-height: 1.5;
}

.hero-classified-text strong {
    color: rgba(220, 210, 255, 0.85);
}

/* ── Mobile Archive Summary (visible only on mobile) ── */
.mobile-archive-summary {
    display: none;
    margin: 1.5rem auto;
    max-width: 700px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(139, 92, 246, 0.12);
    border-radius: 12px;
    background: rgba(15, 15, 35, 0.5);
    backdrop-filter: blur(10px);
}

.mobile-archive-summary .archive-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
}

.mobile-archive-summary .stat-pill {
    font-size: 0.72rem;
    color: rgba(180, 175, 210, 0.7);
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.12);
    border-radius: 20px;
    padding: 0.3rem 0.7rem;
    white-space: nowrap;
}

/* ── Mobile Responsive ── */
@media (max-width: 768px) {
    .hero-container {
        padding: 2rem 1rem 1rem 1rem;
    }
    .hero-glow {
        width: 300px;
        height: 200px;
    }
    .hero-title {
        font-size: 1.7rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        max-width: 100%;
        padding: 0 0.5rem;
    }
    .hero-credential {
        font-size: 0.7rem;
    }
    .hero-classified {
        padding: 0.4rem 0.8rem;
    }
    .hero-classified-text {
        font-size: 0.68rem;
    }
    .mobile-archive-summary {
        display: block;
    }
    .paths-container {
        padding: 0 0.5rem;
    }
    .path-category {
        font-size: 0.65rem;
        margin: 1.2rem 0 0.4rem 0.1rem;
    }
    .stButton > button {
        font-size: 0.82rem !important;
        padding: 0.6rem 0.8rem !important;
    }
    [data-testid="stChatMessage"] {
        padding: 0.8rem !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        font-size: 0.88rem !important;
    }
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    .hero-chip-container .stButton > button {
        font-size: 0.92rem !important;
        padding: 0.8rem 1rem !important;
    }
}

/* ── Exploration Paths ── */
.paths-container {
    max-width: 750px;
    margin: 2.5rem auto 2rem auto;
    position: relative;
    z-index: 1;
}

.path-category {
    font-size: 0.7rem;
    color: rgba(139, 92, 246, 0.55);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin: 1.8rem 0 0.6rem 0.2rem;
}

.path-category:first-child {
    margin-top: 0;
}

/* ── Hero Chip ── */
.hero-chip-container {
    max-width: 750px;
    margin: 1.5rem auto 0.5rem auto;
    position: relative;
    z-index: 1;
}

/* Style the Streamlit buttons as exploration paths */
.stButton > button {
    background: rgba(139, 92, 246, 0.06) !important;
    border: 1px solid rgba(139, 92, 246, 0.15) !important;
    border-radius: 12px !important;
    color: rgba(210, 205, 235, 0.85) !important;
    padding: 0.8rem 1.2rem !important;
    font-size: 0.92rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    backdrop-filter: blur(10px) !important;
    width: 100% !important;
    font-family: 'Inter', sans-serif !important;
}

.stButton > button:hover {
    background: rgba(139, 92, 246, 0.14) !important;
    border-color: rgba(139, 92, 246, 0.35) !important;
    color: #e2e0f0 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.12),
                0 0 15px rgba(139, 92, 246, 0.06) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Hero Chip (emphasized top button) ── */
.hero-chip-container .stButton > button {
    background: rgba(139, 92, 246, 0.12) !important;
    border: 1px solid rgba(139, 92, 246, 0.3) !important;
    font-size: 1.05rem !important;
    padding: 1rem 1.5rem !important;
    color: rgba(230, 225, 250, 0.95) !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.1),
                0 0 30px rgba(139, 92, 246, 0.05) !important;
}

.hero-chip-container .stButton > button:hover {
    background: rgba(139, 92, 246, 0.2) !important;
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 8px 40px rgba(139, 92, 246, 0.18),
                0 0 40px rgba(139, 92, 246, 0.08) !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    max-width: 800px !important;
    margin: 0 auto !important;
}

[data-testid="stChatInput"] textarea {
    background: rgba(15, 15, 35, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 16px !important;
    color: #e2e0f0 !important;
    font-size: 1rem !important;
    padding: 1rem 1.2rem !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.3s ease !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(139, 92, 246, 0.45) !important;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.1),
                0 0 50px rgba(139, 92, 246, 0.05) !important;
    background: rgba(15, 15, 35, 0.95) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(180, 175, 210, 0.5) !important;
    font-style: italic;
}

/* Submit button in chat input */
[data-testid="stChatInput"] button {
    background: rgba(139, 92, 246, 0.25) !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] button:hover {
    background: rgba(139, 92, 246, 0.4) !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: rgba(15, 15, 35, 0.4) !important;
    border: 1px solid rgba(139, 92, 246, 0.08) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
    padding: 1.2rem !important;
    max-width: 850px !important;
    margin: 0.5rem auto !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: rgba(235, 233, 245, 0.92) !important;
    line-height: 1.7 !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: rgba(240, 235, 255, 0.95) !important;
    font-weight: 600 !important;
}

[data-testid="stChatMessage"] strong {
    color: rgba(220, 210, 255, 0.98) !important;
}

[data-testid="stChatMessage"] blockquote {
    border-left: 2px solid rgba(139, 92, 246, 0.3) !important;
    color: rgba(200, 195, 230, 0.75) !important;
    padding-left: 1rem !important;
}

/* ── Sources Expander ── */
.streamlit-expanderHeader {
    background: rgba(139, 92, 246, 0.06) !important;
    border: 1px solid rgba(139, 92, 246, 0.12) !important;
    border-radius: 10px !important;
    color: rgba(180, 170, 220, 0.7) !important;
    font-size: 0.85rem !important;
}

.streamlit-expanderContent {
    background: rgba(10, 10, 25, 0.5) !important;
    border: 1px solid rgba(139, 92, 246, 0.08) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    color: rgba(139, 92, 246, 0.6) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.35);
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    color: rgba(200, 200, 230, 0.6) !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 0.8rem !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    color: rgba(200, 200, 230, 0.85) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Checkbox styling ── */
[data-testid="stSidebar"] .stCheckbox label span {
    color: rgba(200, 200, 230, 0.6) !important;
    font-size: 0.85rem !important;
}

/* ── Dividers ── */
hr {
    border-color: rgba(139, 92, 246, 0.08) !important;
}

/* ── Fix main content area ── */
.main .block-container {
    max-width: 900px !important;
    padding-top: 1rem !important;
}
</style>
"""

# ── Initialize Services (cached) ───────────────────────────────────────────


@st.cache_resource
def init_clients():
    """Initialize API clients once."""
    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX_NAME)
    return claude, voyage, index


def retrieve_context(query: str, voyage_client, pinecone_index, top_k: int = TOP_K) -> list[dict]:
    """Embed the query and retrieve relevant chunks from Pinecone."""
    result = voyage_client.embed([query], model=VOYAGE_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    contexts = []
    for match in results.matches:
        meta = match.metadata
        contexts.append({
            "text": meta.get("text", ""),
            "explorer_id": meta.get("explorer_id", "Unknown"),
            "session_number": meta.get("session_number", "?"),
            "filename": meta.get("filename", ""),
            "topics": meta.get("topics", ""),
            "score": match.score,
        })

    return contexts


def format_source_label(ctx: dict) -> str:
    """Create a readable source label from context metadata."""
    explorer_id = ctx.get("explorer_id", "Unknown")
    filename = ctx.get("filename", "")
    session_number = ctx.get("session_number", 0)

    # Audio transcripts
    if explorer_id in ("Professional_Seminars", "Quarterly_Tapes"):
        name = str(session_number) if session_number else filename
        for suffix in [".txt", ".pdf"]:
            name = name.replace(suffix, "")
        category = "Professional Seminar" if explorer_id == "Professional_Seminars" else "Quarterly Tape"
        return f"{category}: \"{name}\""

    # Robert Monroe files
    if explorer_id.startswith("RAM"):
        name = filename
        for suffix in [".pdf", "_djvu.txt", " (Transcript)", "(Transcript)"]:
            name = name.replace(suffix, "")
        name = name.strip()
        if explorer_id == "RAM_Gateway":
            return f"Robert Monroe, Gateway Voyage: \"{name}\""
        elif explorer_id == "RAM_Guidelines":
            return f"Robert Monroe, Guidelines: \"{name}\""
        elif explorer_id == "RAM_INSCOM":
            return f"Robert Monroe, INSCOM: \"{name}\""
        else:
            return f"Robert Monroe: \"{name}\""

    # Explorer Sessions
    if session_number and session_number != 0:
        return f"Explorer {explorer_id}, Session {session_number}"
    else:
        name = filename.replace(".pdf", "").replace(".txt", "").strip()
        return f"Explorer {explorer_id}: \"{name}\""


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


def build_claude_messages(query: str, contexts: list[dict], chat_history: list[dict]) -> list[dict]:
    """Build the messages list for Claude."""
    context_prompt = build_context_prompt(contexts)

    user_message = f"""Based on the following transcript excerpts from the Monroe Institute archives, please answer the user's question.

{context_prompt}

User's question: {query}"""

    messages = []
    for msg in chat_history[-10:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    return messages


def stream_claude_response(query: str, contexts: list[dict], chat_history: list[dict],
                           claude_client):
    """Stream response from Claude for a conversational feel."""
    messages = build_claude_messages(query, contexts, chat_history)

    with claude_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def handle_query(prompt, claude_client, voyage_client, pinecone_index, top_k, show_sources):
    """Process a user query: retrieve context, stream response, show sources."""
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching the archives..."):
            contexts = retrieve_context(prompt, voyage_client, pinecone_index, top_k=top_k)

        response = st.write_stream(
            stream_claude_response(
                prompt, contexts, st.session_state.chat_history, claude_client
            )
        )

        if contexts and show_sources:
            with st.expander("View source transcripts"):
                for src in contexts:
                    st.markdown(f"**{format_source_label(src)}** "
                                f"(relevance: {src['score']:.0%})")
                    st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": contexts,
    })
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": response})


# ── Streamlit UI ───────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Monroe Institute Archives",
        page_icon="🔮",
        layout="wide",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar (minimal, elegant) ──
    with st.sidebar:
        st.markdown("### Monroe Archives")
        st.caption("Search 770+ transcripts from the Monroe Institute's consciousness research.")

        st.divider()

        with st.expander("What's in the archive?", expanded=False):
            st.markdown("""
**Explorer Sessions** — 460+ sessions
Consciousness explorers navigating focus levels, entity contact, OBEs, and beyond.

**Robert Monroe** — 280+ recordings
Gateway Voyage, Guidelines, INSCOM military sessions, interviews.

**Professional Seminars** — 16 presentations
Researchers on Hemi-Sync, consciousness, and healing.

**Quarterly Tapes** — 14 recordings
Member-exclusive talks and interviews.
            """)

        st.divider()

        show_sources = st.checkbox("Show source citations", value=True)

        with st.expander("Advanced", expanded=False):
            top_k = st.slider(
                "Search depth",
                min_value=3, max_value=15, value=8,
                help="Number of transcript excerpts to retrieve per question"
            )

        st.divider()

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.pop("pending_question", None)
            st.rerun()

    # ── Initialize ──
    try:
        claude_client, voyage_client, pinecone_index = init_clients()
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        st.info("Check your API keys.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── Handle Pending Question (from chip click) ──
    pending = st.session_state.pop("pending_question", None)

    # ── Welcome Portal (before any conversation and no pending) ──
    if not st.session_state.messages and not pending:
        # Hero section
        st.markdown("""
        <div class="hero-container">
            <div class="hero-glow"></div>
            <div class="hero-title">Explore the Edges of<br>Human Consciousness</div>
            <div class="hero-subtitle">
                Search real transcripts from decades of consciousness exploration
                at the Monroe Institute — out-of-body experiences, Focus levels,
                entity contact, and what lies beyond physical reality.
            </div>
            <div class="hero-credential">
                770+ archived sessions · Monroe Institute · 1974–present
            </div>
            <div class="hero-classified">
                <div class="hero-classified-text">
                    Includes material from the <strong>CIA Gateway Process</strong> report
                    and <strong>U.S. Army INSCOM</strong> remote viewing sessions —
                    formerly classified, now declassified
                </div>
            </div>
        </div>
        <div class="mobile-archive-summary">
            <div class="archive-stats">
                <span class="stat-pill">460+ Explorer Sessions</span>
                <span class="stat-pill">280+ Robert Monroe Talks</span>
                <span class="stat-pill">CIA Gateway Report</span>
                <span class="stat-pill">INSCOM Military Sessions</span>
                <span class="stat-pill">16 Professional Seminars</span>
                <span class="stat-pill">14 Quarterly Tapes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hero chip (emphasized)
        st.markdown('<div class="hero-chip-container">', unsafe_allow_html=True)
        hero_label = f"{HERO_CHIP['icon']}  {HERO_CHIP['label']}"
        if st.button(hero_label, key="hero_chip", use_container_width=True):
            st.session_state.pending_question = HERO_CHIP["label"]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Exploration paths
        st.markdown('<div class="paths-container">', unsafe_allow_html=True)

        chip_idx = 0
        for category, paths in EXPLORATION_PATHS.items():
            st.markdown(f'<div class="path-category">{category}</div>',
                        unsafe_allow_html=True)
            cols = st.columns(2)
            for i, path in enumerate(paths):
                col = cols[i % 2]
                btn_label = f"{path['icon']}  {path['label']}"
                if col.button(btn_label, key=f"chip_{chip_idx}",
                              use_container_width=True):
                    st.session_state.pending_question = path["label"]
                    st.rerun()
                chip_idx += 1

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── Chat History ──
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources") and show_sources:
                    with st.expander("View source transcripts"):
                        for src in message["sources"]:
                            st.markdown(f"**{format_source_label(src)}** "
                                        f"(relevance: {src['score']:.0%})")
                            st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                            st.divider()

    # ── Handle Pending Chip Question ──
    if pending:
        with st.chat_message("user"):
            st.markdown(pending)
        handle_query(pending, claude_client, voyage_client, pinecone_index, top_k, show_sources)

    # ── Chat Input ──
    if prompt := st.chat_input("Ask about out-of-body experiences, Focus levels, or what explorers discovered..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_query(prompt, claude_client, voyage_client, pinecone_index, top_k, show_sources)


if __name__ == "__main__":
    main()
