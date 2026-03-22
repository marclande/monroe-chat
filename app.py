"""
Monroe Institute Explorer Sessions — Chat Interface
=====================================================
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

SYSTEM_PROMPT = """You are an expert research assistant specializing in the Monroe Institute's Explorer Sessions — a collection of documented consciousness exploration sessions conducted primarily between 1974 and 1989 under the guidance of Robert Monroe.

You have deep knowledge of:
- Focus levels (Focus 10, 12, 15, 21, etc.) and their characteristics
- Hemi-Sync technology and its applications
- The various explorers (identified by codes like IMEC, SHE, GLA, AUB, etc.) and their unique experiences
- Channeled entities and communications (e.g., Miranon through SHE/Shay Ellsworth)
- Topics including: astral travel, entity communication, healing, consciousness states, past lives, death/dying, spirit rescue, time/space perception, energy phenomena, and planetary consciousness

INSTRUCTIONS:
1. Ground your answers in the actual transcript excerpts provided as context. Always cite your sources with the explorer code, session number, and relevant quote.
2. When synthesizing across sessions, note which explorers/sessions contribute to each point.
3. If the context doesn't contain enough information to answer fully, say so honestly — don't fabricate details about the sessions.
4. Use a warm, knowledgeable tone — like a seasoned researcher sharing fascinating findings.
5. When asked about specific explorers, sessions, or phenomena, provide specific details from the transcripts.
6. If a question is about something not covered in the provided context, acknowledge that and offer to search for related topics.

FORMAT:
- Use clear paragraphs for readability
- Include direct quotes from transcripts when relevant, formatted with quotation marks
- Cite sources as (Explorer CODE, Session #) — e.g., (SHE, Session 6)
- For cross-session themes, organize your response thematically"""

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
    # Embed the query
    result = voyage_client.embed([query], model=VOYAGE_MODEL, input_type="query")
    query_embedding = result.embeddings[0]

    # Query Pinecone
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    # Format results
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


def build_context_prompt(contexts: list[dict]) -> str:
    """Format retrieved contexts into a prompt section."""
    if not contexts:
        return "No relevant transcript excerpts were found for this query."

    parts = ["Here are the most relevant transcript excerpts:\n"]
    for i, ctx in enumerate(contexts, 1):
        parts.append(f"--- Excerpt {i} (Explorer: {ctx['explorer_id']}, "
                     f"Session: {ctx['session_number']}, "
                     f"Relevance: {ctx['score']:.2f}) ---")
        parts.append(ctx["text"])
        if ctx["topics"]:
            parts.append(f"[Topics: {ctx['topics']}]")
        parts.append("")

    return "\n".join(parts)


def chat_with_claude(query: str, contexts: list[dict], chat_history: list[dict],
                     claude_client) -> str:
    """Send query + context + history to Claude and get a response."""
    context_prompt = build_context_prompt(contexts)

    # Build message with context
    user_message = f"""Based on the following transcript excerpts from the Monroe Institute Explorer Sessions, please answer the user's question.

{context_prompt}

User's question: {query}"""

    # Build messages list with history
    messages = []
    for msg in chat_history[-10:]:  # Keep last 10 messages for context
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return response.content[0].text


# ── Streamlit UI ───────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Monroe Institute Explorer",
        page_icon="🔮",
        layout="wide",
    )

    # Header
    st.title("🔮 Monroe Institute Explorer Sessions")
    st.caption("Ask questions about consciousness exploration sessions from 1974–1989. "
               "Powered by RAG over 460+ session transcripts.")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app searches across **460+ transcripts** from the Monroe Institute's
        Explorer Sessions and uses Claude to synthesize answers grounded in the
        actual session content.

        **Explorers include:**
        - IMEC (Marie Coble)
        - SHE (Shay Ellsworth / Miranon)
        - GLA, AUB, RPE, MDG, and 50+ others

        **Topics covered:**
        - Focus levels & Hemi-Sync
        - Entity communication
        - Astral travel & OBEs
        - Healing & energy work
        - Past lives & reincarnation
        - Death, dying & spirit rescue
        - Time/space perception
        - Prophecies & Earth changes
        """)

        st.divider()

        st.header("Settings")
        top_k = st.slider("Sources to retrieve", min_value=3, max_value=15, value=8)
        show_sources = st.checkbox("Show source excerpts", value=True)

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

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

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources") and show_sources:
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.markdown(f"**{src['explorer_id']} — Session {src['session_number']}** "
                                    f"(relevance: {src['score']:.0%})")
                        st.caption(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask about the Explorer Sessions..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve and respond
        with st.chat_message("assistant"):
            with st.spinner("Searching transcripts..."):
                contexts = retrieve_context(prompt, voyage_client, pinecone_index, top_k=top_k)

            with st.spinner("Synthesizing answer..."):
                response = chat_with_claude(
                    prompt, contexts, st.session_state.chat_history, claude_client
                )

            st.markdown(response)

            # Show sources
            if contexts and show_sources:
                with st.expander("📚 Sources"):
                    for src in contexts:
                        st.markdown(f"**{src['explorer_id']} — Session {src['session_number']}** "
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


if __name__ == "__main__":
    main()
