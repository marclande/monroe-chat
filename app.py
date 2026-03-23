"""
Monroe Institute Archives — Chat Interface
=============================================
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
TOP_K = 8  # Number of chunks to retrieve

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

# Example questions for new users
EXAMPLE_QUESTIONS = [
    "What is Focus 15 and how do explorers experience it?",
    "What did Robert Monroe say about out-of-body experiences?",
    "Tell me about Miranon's teachings through explorer SHE",
    "What role does the monitor play in Explorer sessions?",
    "What did the explorers discover about death and the afterlife?",
    "How does Hemi-Sync technology work according to Monroe?",
]

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

    # Audio transcripts (Professional Seminars, Quarterly Tapes)
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
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve and respond
    with st.chat_message("assistant"):
        with st.spinner("Searching archives..."):
            contexts = retrieve_context(prompt, voyage_client, pinecone_index, top_k=top_k)

        # Stream the response
        response = st.write_stream(
            stream_claude_response(
                prompt, contexts, st.session_state.chat_history, claude_client
            )
        )

        # Show sources
        if contexts and show_sources:
            with st.expander("📚 Sources"):
                for src in contexts:
                    st.markdown(f"**{format_source_label(src)}** "
                                f"(relevance: {src['score']:.0%})")
                    st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                    st.divider()

    # Save to state
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

    # Custom CSS for cleaner look
    st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .example-btn { margin: 2px 0; }
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar — concise
    with st.sidebar:
        st.markdown("### 🔮 Monroe Archives")
        st.caption(
            "AI-powered search across 770+ transcripts from the Monroe Institute's "
            "consciousness research archives (1974–present)."
        )

        st.divider()

        with st.expander("📖 What's in the archive?", expanded=False):
            st.markdown("""
**Explorer Sessions** — 460+ sessions (1974–1990)
Consciousness explorers like IMEC, SHE, GLA, AUB, and 50+ others exploring focus levels, entity communication, OBEs, and more.

**Robert Monroe** — 280+ talks & interviews
Gateway Voyage talks, Guidelines, INSCOM military sessions, and personal interviews.

**Professional Seminars** — 16 presentations
Skip Atwater, Beverly Rubik, Rita Warren, and other researchers on Hemi-Sync applications.

**Quarterly Member Tapes** — 14 recordings
Member-exclusive talks, interviews, and discussions.
            """)

        st.divider()

        # Settings — simplified
        show_sources = st.checkbox("Show source citations", value=True)

        with st.expander("⚙️ Advanced", expanded=False):
            top_k = st.slider(
                "Search depth",
                min_value=3, max_value=15, value=8,
                help="How many transcript excerpts to search through for each question"
            )

        st.divider()
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.pop("pending_question", None)
            st.rerun()

        st.divider()
        st.caption("Built with Claude, Voyage AI & Pinecone")

    # Initialize clients
    try:
        claude_client, voyage_client, pinecone_index = init_clients()
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        st.info("Check your API keys in the .env file.")
        return

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Welcome screen (only when no messages yet)
    if not st.session_state.messages:
        st.markdown("## 🔮 Monroe Institute Archives")
        st.markdown(
            "Ask anything about the Monroe Institute's consciousness research — "
            "Explorer sessions, Robert Monroe's talks, Focus levels, Hemi-Sync, "
            "out-of-body experiences, and more."
        )
        st.markdown("")
        st.markdown("**Try one of these questions:**")

        # Example question buttons in 2 columns
        cols = st.columns(2)
        for i, question in enumerate(EXAMPLE_QUESTIONS):
            col = cols[i % 2]
            if col.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()
    else:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources") and show_sources:
                    with st.expander("📚 Sources"):
                        for src in message["sources"]:
                            st.markdown(f"**{format_source_label(src)}** "
                                        f"(relevance: {src['score']:.0%})")
                            st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                            st.divider()

    # Handle pending question from example buttons
    if "pending_question" in st.session_state:
        pending = st.session_state.pop("pending_question")
        with st.chat_message("user"):
            st.markdown(pending)
        handle_query(pending, claude_client, voyage_client, pinecone_index, top_k, show_sources)
        st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about the Monroe Institute archives..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_query(prompt, claude_client, voyage_client, pinecone_index, top_k, show_sources)


if __name__ == "__main__":
    main()
