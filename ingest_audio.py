"""
Ingest Audio Transcripts into Pinecone
=======================================
Reads Whisper-generated .txt transcripts from the audio archives
and adds them to the existing Pinecone index.
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

import voyageai
from pinecone import Pinecone
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────────

AUDIO_TRANSCRIPTS_DIR = Path("C:/Users/marc.lande/Downloads/monroe_archive/monroe_archive/transcripts_audio")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "monroe-explorer")

CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
VOYAGE_MODEL = "voyage-3"
VOYAGE_BATCH_SIZE = 100
EMBEDDING_DIM = 1024

# ── Helpers ─────────────────────────────────────────────────────────────────

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def chunk_text(text: str, target_tokens: int = CHUNK_TARGET_TOKENS,
               overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    import re
    # Split on sentences for Whisper output (no paragraph breaks)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens > target_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep last 1-2 sentences as overlap
            overlap = []
            overlap_tok = 0
            for s in reversed(current_chunk):
                st = count_tokens(s)
                if overlap_tok + st <= overlap_tokens:
                    overlap.insert(0, s)
                    overlap_tok += st
                else:
                    break
            current_chunk = overlap
            current_tokens = overlap_tok
        current_chunk.append(sent)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def make_chunk_id(source: str, chunk_index: int) -> str:
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def run_ingestion():
    print("=" * 70)
    print("Monroe Institute Audio Transcripts - Ingestion Pipeline")
    print("=" * 70)

    # Connect to services
    print("\nConnecting to Voyage AI...")
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX_NAME)

    stats = index.describe_index_stats()
    print(f"  Current vectors in index: {stats.total_vector_count}")

    # Find all audio transcript .txt files
    txt_files = sorted(AUDIO_TRANSCRIPTS_DIR.rglob("*.txt"))
    print(f"\nFound {len(txt_files)} audio transcript files")

    all_chunks = []
    all_metadata = []

    for i, txt_path in enumerate(txt_files):
        category = txt_path.parent.name
        filename = txt_path.stem
        print(f"  [{i+1}/{len(txt_files)}] {category}/{filename}...", end=" ")

        text = txt_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            print("empty, skipping")
            continue

        chunks = chunk_text(text)
        print(f"{len(chunks)} chunks ({count_tokens(text)} tokens)")

        for ci, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(f"audio_{category}_{filename}", ci)
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "category": category,
                    "content_type": "audio_transcript",
                    "explorer_id": category,
                    "session_number": filename,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                    "text_preview": chunk[:1000],
                }
            })

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # Embed and upsert
    print(f"\nEmbedding with Voyage AI ({VOYAGE_MODEL})...")

    vectors_upserted = 0
    for batch_start in range(0, len(all_chunks), VOYAGE_BATCH_SIZE):
        batch_end = min(batch_start + VOYAGE_BATCH_SIZE, len(all_chunks))
        batch = all_chunks[batch_start:batch_end]
        texts = [c["text"] for c in batch]

        print(f"  Batch {batch_start}-{batch_end}...", end=" ")

        for attempt in range(3):
            try:
                result = vo.embed(texts, model=VOYAGE_MODEL, input_type="document")
                embeddings = result.embeddings

                vectors = []
                for j, emb in enumerate(embeddings):
                    c = batch[j]
                    vectors.append({
                        "id": c["id"],
                        "values": emb,
                        "metadata": c["metadata"],
                    })

                index.upsert(vectors=vectors)
                vectors_upserted += len(vectors)
                print(f"done ({len(vectors)} vectors)")
                break
            except Exception as e:
                wait = 22 * (attempt + 1)
                print(f"\n  Rate limited, waiting {wait}s... (attempt {attempt+1}/3)")
                time.sleep(wait)
        else:
            print(f"\n  Failed after 3 attempts on batch {batch_start}-{batch_end}")

        time.sleep(1)  # Small delay between batches

    # Final stats
    stats = index.describe_index_stats()
    print(f"\n{'=' * 70}")
    print(f"INGESTION COMPLETE")
    print(f"  Audio transcripts processed: {len(txt_files)}")
    print(f"  Chunks embedded: {len(all_chunks)}")
    print(f"  Vectors upserted: {vectors_upserted}")
    print(f"  Total vectors in Pinecone: {stats.total_vector_count}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_ingestion()
