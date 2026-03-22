"""
Monroe Institute Explorer Sessions — Ingestion Pipeline
========================================================
Extracts text from PDFs, chunks semantically, embeds with Voyage AI,
and stores in Pinecone for RAG retrieval.
"""

import os
import re
import sys
import json
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

import fitz  # PyMuPDF
import voyageai
from pinecone import Pinecone, ServerlessSpec
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────────

TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR"))
METADATA_PATH = Path(os.getenv("METADATA_PATH"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "monroe-explorer")

CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
VOYAGE_MODEL = "voyage-3"
VOYAGE_BATCH_SIZE = 100  # Maximize texts per API call
VOYAGE_RATE_LIMIT_DELAY = 1  # Minimal delay with payment method (standard rate limits)
EMBEDDING_DIM = 1024

# ── Helpers ─────────────────────────────────────────────────────────────────

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  ⚠ Error extracting {pdf_path.name}: {e}")
        return ""


def extract_text_from_txt(txt_path: Path) -> str:
    """Read a plain text file."""
    try:
        return txt_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  ⚠ Error reading {txt_path.name}: {e}")
        return ""


def chunk_text(text: str, target_tokens: int = CHUNK_TARGET_TOKENS,
               overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """
    Split text into chunks of ~target_tokens with overlap.
    Respects paragraph boundaries where possible.
    """
    if not text.strip():
        return []

    # Split into paragraphs (double newline or more)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If a single paragraph exceeds target, split by sentences
        if para_tokens > target_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > target_tokens and current_chunk:
                    chunk_text_str = "\n\n".join(current_chunk)
                    chunks.append(chunk_text_str)

                    # Overlap: keep last portion
                    overlap_text = chunk_text_str
                    overlap_toks = count_tokens(overlap_text)
                    while overlap_toks > overlap_tokens and "\n\n" in overlap_text:
                        overlap_text = overlap_text.split("\n\n", 1)[1]
                        overlap_toks = count_tokens(overlap_text)
                    current_chunk = [overlap_text] if overlap_toks <= overlap_tokens else []
                    current_tokens = count_tokens(" ".join(current_chunk))

                current_chunk.append(sent)
                current_tokens += sent_tokens
        elif current_tokens + para_tokens > target_tokens and current_chunk:
            chunk_text_str = "\n\n".join(current_chunk)
            chunks.append(chunk_text_str)

            # Overlap
            overlap_text = chunk_text_str
            overlap_toks = count_tokens(overlap_text)
            while overlap_toks > overlap_tokens and "\n\n" in overlap_text:
                overlap_text = overlap_text.split("\n\n", 1)[1]
                overlap_toks = count_tokens(overlap_text)
            current_chunk = [overlap_text] if overlap_toks <= overlap_tokens else []
            current_tokens = count_tokens(" ".join(current_chunk))

            current_chunk.append(para)
            current_tokens += para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def make_chunk_id(explorer_id: str, filename: str, chunk_index: int) -> str:
    """Create a deterministic, unique ID for each chunk."""
    raw = f"{explorer_id}:{filename}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Load Metadata ──────────────────────────────────────────────────────────

def load_metadata() -> dict:
    """Load session metadata keyed by filename for easy lookup."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    by_filename = {}
    for s in sessions:
        # Normalize filename for matching
        fname = s.get("filename", "").strip()
        if fname.startswith("*"):
            fname = fname[1:]
        by_filename[fname.lower()] = s

    return by_filename


def find_metadata_for_file(filename: str, explorer_id: str, metadata: dict) -> dict:
    """Try to match a transcript file to its metadata entry."""
    # Direct match
    key = filename.lower()
    if key in metadata:
        return metadata[key]

    # Try without (Transcript) suffix
    simplified = key.replace(" (transcript)", "").replace("(transcript)", "")
    for mkey, mval in metadata.items():
        if simplified in mkey or mkey in simplified:
            return mval

    # Return basic info
    return {"explorer_id": explorer_id, "topics": [], "session_number": 0}


# ── Main Pipeline ──────────────────────────────────────────────────────────

def run_ingestion():
    print("=" * 70)
    print("Monroe Institute Explorer Sessions — Ingestion Pipeline")
    print("=" * 70)

    # ── Step 1: Initialize services ──
    print("\n📡 Connecting to Voyage AI...")
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    print("📡 Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Create or connect to index
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"  Creating index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIM})...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
        # Wait for index to be ready
        print("  Waiting for index to be ready...")
        time.sleep(10)
    else:
        print(f"  Index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)

    # ── Step 2: Load metadata ──
    print("\n📋 Loading session metadata...")
    metadata = load_metadata()
    print(f"  {len(metadata)} sessions in metadata")

    # ── Step 3: Discover all transcript files ──
    print(f"\n📂 Scanning {TRANSCRIPTS_DIR} for transcripts...")
    all_files = []
    for explorer_dir in sorted(TRANSCRIPTS_DIR.iterdir()):
        if not explorer_dir.is_dir():
            continue
        explorer_id = explorer_dir.name
        for f in sorted(explorer_dir.iterdir()):
            if f.suffix.lower() in (".pdf", ".txt"):
                all_files.append((explorer_id, f))

    print(f"  Found {len(all_files)} transcript files")

    # ── Step 4: Extract, chunk, embed, and upsert ──
    print("\n🔄 Processing transcripts...\n")

    total_chunks = 0
    total_files_processed = 0
    failed_files = []

    # Collect all chunks first, then batch-embed
    all_chunks_data = []

    for i, (explorer_id, filepath) in enumerate(all_files):
        print(f"  [{i + 1}/{len(all_files)}] {explorer_id}/{filepath.name}...", end=" ")

        # Extract text
        if filepath.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)

        if not text or len(text.strip()) < 50:
            print("⚠ No text extracted (possibly scanned/image PDF)")
            failed_files.append(str(filepath))
            continue

        # Chunk
        chunks = chunk_text(text)
        if not chunks:
            print("⚠ No chunks produced")
            failed_files.append(str(filepath))
            continue

        # Look up metadata
        meta = find_metadata_for_file(filepath.stem, explorer_id, metadata)
        topics = meta.get("topics", [])
        session_number = meta.get("session_number", 0)

        for ci, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(explorer_id, filepath.name, ci)
            all_chunks_data.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "explorer_id": explorer_id,
                    "session_number": session_number,
                    "filename": filepath.name,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                    "topics": ", ".join(topics) if topics else "",
                    "text": chunk[:1000],  # Store truncated text in metadata for retrieval display
                },
            })

        total_files_processed += 1
        total_chunks += len(chunks)
        print(f"✓ {len(chunks)} chunks ({count_tokens(text)} tokens)")

    print(f"\n{'=' * 70}")
    print(f"Extraction complete: {total_files_processed} files → {total_chunks} chunks")
    if failed_files:
        print(f"⚠ {len(failed_files)} files failed extraction")

    # ── Step 5: Batch embed with Voyage AI ──
    print(f"\n🧠 Embedding {len(all_chunks_data)} chunks with Voyage AI ({VOYAGE_MODEL})...")

    texts_to_embed = [c["text"] for c in all_chunks_data]
    all_embeddings = []

    for batch_start in range(0, len(texts_to_embed), VOYAGE_BATCH_SIZE):
        batch_end = min(batch_start + VOYAGE_BATCH_SIZE, len(texts_to_embed))
        batch = texts_to_embed[batch_start:batch_end]

        # Retry up to 3 times with increasing delays for rate limiting
        for attempt in range(3):
            try:
                result = vo.embed(batch, model=VOYAGE_MODEL, input_type="document")
                all_embeddings.extend(result.embeddings)
                batch_num = batch_start // VOYAGE_BATCH_SIZE + 1
                total_batches = (len(texts_to_embed) + VOYAGE_BATCH_SIZE - 1) // VOYAGE_BATCH_SIZE
                print(f"  Embedded batch {batch_num}/{total_batches} "
                      f"(chunks {batch_start + 1}-{batch_end} / {len(texts_to_embed)})")
                break
            except Exception as e:
                wait = VOYAGE_RATE_LIMIT_DELAY * (attempt + 1)
                if attempt < 2:
                    print(f"  Rate limited, waiting {wait}s... (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                else:
                    print(f"  ✗ Failed after 3 attempts on batch {batch_start}-{batch_end}")
                    all_embeddings.extend([None] * len(batch))

        # Respect rate limit: 3 RPM = wait 22s between calls
        time.sleep(VOYAGE_RATE_LIMIT_DELAY)

    # ── Step 6: Upsert to Pinecone ──
    print(f"\n📤 Upserting {len(all_chunks_data)} vectors to Pinecone...")

    UPSERT_BATCH_SIZE = 100
    upserted = 0

    for batch_start in range(0, len(all_chunks_data), UPSERT_BATCH_SIZE):
        batch_end = min(batch_start + UPSERT_BATCH_SIZE, len(all_chunks_data))
        vectors = []

        for j in range(batch_start, batch_end):
            if all_embeddings[j] is not None:
                vectors.append({
                    "id": all_chunks_data[j]["id"],
                    "values": all_embeddings[j],
                    "metadata": all_chunks_data[j]["metadata"],
                })

        if vectors:
            try:
                index.upsert(vectors=vectors)
                upserted += len(vectors)
                print(f"  Upserted {upserted} / {len(all_chunks_data)} vectors")
            except Exception as e:
                print(f"  ⚠ Upsert error at batch {batch_start}: {e}")

    # ── Done ──
    print(f"\n{'=' * 70}")
    print(f"✅ INGESTION COMPLETE")
    print(f"   Files processed: {total_files_processed}")
    print(f"   Total chunks:    {total_chunks}")
    print(f"   Vectors upserted: {upserted}")
    print(f"   Pinecone index:  {PINECONE_INDEX_NAME}")
    if failed_files:
        print(f"\n⚠ Failed files ({len(failed_files)}):")
        for f in failed_files[:20]:
            print(f"   - {f}")
        if len(failed_files) > 20:
            print(f"   ... and {len(failed_files) - 20} more")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_ingestion()
