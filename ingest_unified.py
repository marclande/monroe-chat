"""
Monroe Institute — Unified Ingestion Pipeline v2
==================================================
Consolidates all three ingestion scripts with improvements:
1. Text cleaning (Whisper artifacts, timestamps, repetitions)
2. Larger chunks (700 tokens target, topic-aware boundaries)
3. BM25 corpus export for hybrid search
"""

import os
import re
import sys
import json
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

import fitz  # PyMuPDF
import voyageai
from pinecone import Pinecone, ServerlessSpec
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────────

TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR",
    "C:/Users/marc.lande/Downloads/monroe_archive/monroe_archive/transcripts"))
EXPANDED_DIR = Path("C:/Users/marc.lande/Downloads/monroe_archive/monroe_archive/transcripts_expanded")
AUDIO_DIR = Path("C:/Users/marc.lande/Downloads/monroe_archive/monroe_archive/transcripts_audio")
METADATA_PATH = Path(os.getenv("METADATA_PATH",
    "C:/Users/marc.lande/Downloads/monroe_archive/monroe_archive/metadata/all_sessions.json"))
BM25_CORPUS_PATH = Path("C:/Users/marc.lande/monroe-chat/bm25_corpus.json")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "monroe-explorer")

CHUNK_TARGET_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 100
VOYAGE_MODEL = "voyage-3"
VOYAGE_BATCH_SIZE = 100
EMBEDDING_DIM = 1024

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


# ── Text Cleaning ───────────────────────────────────────────────────────────

def clean_whisper_text(text: str) -> str:
    """Clean Whisper-generated transcripts: remove repetitions and timestamps."""
    # Remove inline timestamps like "14:01" or "31:53" at start of lines
    text = re.sub(r'^\d{1,2}:\d{2}\s*', '', text, flags=re.MULTILINE)
    # Remove "Speaker HH:MM" patterns like "Bob Monroe 14:16"
    text = re.sub(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+\d{1,2}:\d{2}\b', r'\1', text)

    # Collapse repeated sentences (Whisper hallucination artifact)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    prev = None
    repeat_count = 0
    for sent in sentences:
        sent_stripped = sent.strip()
        if sent_stripped == prev:
            repeat_count += 1
            if repeat_count < 2:  # Keep max 2 occurrences
                cleaned.append(sent_stripped)
        else:
            cleaned.append(sent_stripped)
            prev = sent_stripped
            repeat_count = 0

    text = " ".join(cleaned)

    # Collapse excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_djvu_text(text: str) -> str:
    """Clean OCR'd _djvu.txt transcripts: strip timestamp lines, preserve speaker labels."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove standalone timestamp lines like "SHE 02:16" or "Bob Monroe 01:52"
        if re.match(r'^[A-Z][A-Za-z\s]+\d{1,2}:\d{2}\s*$', stripped):
            # Convert to speaker label: "SHE 02:16" -> "SHE:"
            speaker = re.sub(r'\s*\d{1,2}:\d{2}\s*$', '', stripped)
            cleaned_lines.append(f"\n{speaker}:")
        # Remove pure timestamp lines like "(0:00)" or "(13:08)"
        elif re.match(r'^\(\d{1,2}:\d{2}\)\s*$', stripped):
            continue
        # Remove Otter.ai boilerplate
        elif 'otter.ai' in stripped.lower() or 'automatically generated' in stripped.lower():
            continue
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def clean_general_text(text: str) -> str:
    """General text cleanup for all sources."""
    # Remove excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    # Remove page numbers
    text = re.sub(r'\n\s*-?\s*\d{1,3}\s*-?\s*\n', '\n', text)
    return text.strip()


# ── Improved Chunking ───────────────────────────────────────────────────────

def chunk_text(text: str, target_tokens: int = CHUNK_TARGET_TOKENS,
               overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """
    Split text into chunks of ~target_tokens with overlap.
    Uses a hierarchy of split points: paragraphs > speaker turns > sentences.
    Target: 700 tokens, range 500-900.
    """
    if not text.strip():
        return []

    # Split into segments (paragraphs, speaker turns)
    # Speaker turns are lines starting with a name followed by colon
    segments = re.split(r'\n\s*\n', text)
    segments = [s.strip() for s in segments if s.strip()]

    # Further split large segments on speaker boundaries
    expanded = []
    for seg in segments:
        # Check if segment contains speaker turns
        speaker_splits = re.split(r'\n(?=[A-Z][A-Za-z\s]*:)', seg)
        if len(speaker_splits) > 1:
            expanded.extend([s.strip() for s in speaker_splits if s.strip()])
        else:
            expanded.append(seg)
    segments = expanded

    chunks = []
    current_parts = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = count_tokens(seg)

        # If a single segment exceeds target, split by sentences
        if seg_tokens > target_tokens:
            # Flush current chunk first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = _get_overlap(current_parts, overlap_tokens)
                current_tokens = count_tokens("\n\n".join(current_parts)) if current_parts else 0

            sentences = re.split(r'(?<=[.!?])\s+', seg)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > target_tokens and current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = _get_overlap(current_parts, overlap_tokens)
                    current_tokens = count_tokens("\n\n".join(current_parts)) if current_parts else 0
                current_parts.append(sent)
                current_tokens += sent_tokens

        elif current_tokens + seg_tokens > target_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = _get_overlap(current_parts, overlap_tokens)
            current_tokens = count_tokens("\n\n".join(current_parts)) if current_parts else 0
            current_parts.append(seg)
            current_tokens += seg_tokens
        else:
            current_parts.append(seg)
            current_tokens += seg_tokens

    if current_parts:
        chunk_text_str = "\n\n".join(current_parts)
        # Only add if it has meaningful content (at least 50 tokens)
        if count_tokens(chunk_text_str) >= 50:
            chunks.append(chunk_text_str)

    return chunks


def _get_overlap(parts: list[str], overlap_tokens: int) -> list[str]:
    """Get the last N tokens worth of parts for overlap."""
    if not parts:
        return []
    overlap = []
    tok_count = 0
    for part in reversed(parts):
        pt = count_tokens(part)
        if tok_count + pt <= overlap_tokens:
            overlap.insert(0, part)
            tok_count += pt
        else:
            break
    return overlap


# ── Text Extraction ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  Error extracting {pdf_path.name}: {e}")
        return ""


def extract_text_from_txt(txt_path: Path) -> str:
    try:
        return txt_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  Error reading {txt_path.name}: {e}")
        return ""


def make_chunk_id(source: str, filename: str, chunk_index: int) -> str:
    raw = f"v2:{source}:{filename}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Metadata ────────────────────────────────────────────────────────────────

def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        sessions = json.load(f)
    by_filename = {}
    for s in sessions:
        fname = s.get("filename", "").strip()
        if fname.startswith("*"):
            fname = fname[1:]
        by_filename[fname.lower()] = s
    return by_filename


def find_metadata_for_file(filename: str, explorer_id: str, metadata: dict) -> dict:
    key = filename.lower()
    if key in metadata:
        return metadata[key]
    simplified = key.replace(" (transcript)", "").replace("(transcript)", "")
    for mkey, mval in metadata.items():
        if simplified in mkey or mkey in simplified:
            return mval
    return {"explorer_id": explorer_id, "topics": [], "session_number": 0}


# ── Source Discovery ────────────────────────────────────────────────────────

def discover_explorer_sessions() -> list[tuple]:
    """Original Explorer Session transcripts (PDFs + TXTs)."""
    files = []
    if not TRANSCRIPTS_DIR.exists():
        return files
    for explorer_dir in sorted(TRANSCRIPTS_DIR.iterdir()):
        if not explorer_dir.is_dir():
            continue
        explorer_id = explorer_dir.name
        for f in sorted(explorer_dir.iterdir()):
            if f.suffix.lower() in (".pdf", ".txt"):
                files.append(("explorer", explorer_id, f))
    return files


def discover_expanded_transcripts() -> list[tuple]:
    """RAM expanded transcripts (Gateway, Guidelines, Interviews, etc.)."""
    files = []
    if not EXPANDED_DIR.exists():
        return files
    for category_dir in sorted(EXPANDED_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for f in sorted(category_dir.iterdir()):
            if f.suffix.lower() == ".pdf" and "(Transcript)" in f.name:
                files.append(("expanded", category, f))
            elif f.name.endswith("_djvu.txt") and "(Transcript)" in f.name:
                pdf_name = f.name.replace("_djvu.txt", ".pdf")
                if not (f.parent / pdf_name).exists():
                    files.append(("expanded", category, f))
    return files


def discover_audio_transcripts() -> list[tuple]:
    """Whisper-generated audio transcripts."""
    files = []
    if not AUDIO_DIR.exists():
        return files
    for txt_path in sorted(AUDIO_DIR.rglob("*.txt")):
        category = txt_path.parent.name
        files.append(("audio", category, txt_path))
    return files


# ── Main Pipeline ───────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("Monroe Institute — Unified Ingestion Pipeline v2")
    print("=" * 70)
    print("Improvements: text cleaning, 700-token chunks, BM25 corpus export")

    # ── Initialize services ──
    print("\nConnecting to Voyage AI...")
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Clear and recreate index
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing:
        print(f"  Deleting existing index '{PINECONE_INDEX_NAME}'...")
        pc.delete_index(PINECONE_INDEX_NAME)
        time.sleep(5)

    print(f"  Creating fresh index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIM})...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        ),
    )
    time.sleep(10)
    index = pc.Index(PINECONE_INDEX_NAME)

    # ── Load metadata ──
    print("\nLoading session metadata...")
    metadata = load_metadata()
    print(f"  {len(metadata)} sessions in metadata")

    # ── Discover all sources ──
    print("\nDiscovering transcript sources...")
    explorer_files = discover_explorer_sessions()
    expanded_files = discover_expanded_transcripts()
    audio_files = discover_audio_transcripts()
    print(f"  Explorer Sessions: {len(explorer_files)} files")
    print(f"  Expanded (RAM):    {len(expanded_files)} files")
    print(f"  Audio Transcripts: {len(audio_files)} files")

    all_source_files = explorer_files + expanded_files + audio_files
    print(f"  Total: {len(all_source_files)} files to process")

    # ── Process all sources ──
    print("\nProcessing transcripts...\n")
    all_chunks_data = []
    bm25_corpus = {}  # chunk_id -> full text
    processed = 0
    failed = []

    for i, (source_type, category, filepath) in enumerate(all_source_files):
        print(f"  [{i+1}/{len(all_source_files)}] {category}/{filepath.name}...", end=" ")

        # Extract text
        if filepath.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)

        if not text or len(text.strip()) < 50:
            print("No text")
            failed.append(str(filepath))
            continue

        # Clean text based on source type
        if source_type == "audio":
            text = clean_whisper_text(text)
        elif filepath.name.endswith("_djvu.txt"):
            text = clean_djvu_text(text)
        text = clean_general_text(text)

        if not text or len(text.strip()) < 50:
            print("No text after cleaning")
            failed.append(str(filepath))
            continue

        # Chunk
        chunks = chunk_text(text)
        if not chunks:
            print("No chunks")
            failed.append(str(filepath))
            continue

        # Build metadata per source type
        if source_type == "explorer":
            meta = find_metadata_for_file(filepath.stem, category, metadata)
            explorer_id = category
            session_number = meta.get("session_number", 0)
            topics = meta.get("topics", [])
            topics_str = ", ".join(topics) if topics else ""
            content_type = "explorer_session"
        elif source_type == "expanded":
            explorer_id = "RAM"
            if "INSCOM" in category:
                explorer_id = "RAM_INSCOM"
            elif "Guidelines" in category:
                explorer_id = "RAM_Guidelines"
            elif "Gateway" in category:
                explorer_id = "RAM_Gateway"
            session_number = 0
            topics_str = f"Robert Monroe, {category.replace('_', ' ')}"
            content_type = "ram_transcript"
        else:  # audio
            explorer_id = category
            session_number = filepath.stem
            topics_str = f"{category.replace('_', ' ')}"
            content_type = "audio_transcript"

        for ci, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(source_type, filepath.name, ci)
            chunk_data = {
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "explorer_id": explorer_id,
                    "session_number": session_number,
                    "filename": filepath.name,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                    "topics": topics_str,
                    "content_type": content_type,
                    "text": chunk[:2000],  # Increased from 1000
                },
            }
            all_chunks_data.append(chunk_data)
            bm25_corpus[chunk_id] = chunk  # Full text for BM25

        processed += 1
        print(f"{len(chunks)} chunks ({count_tokens(text)} tokens)")

    print(f"\n{'='*70}")
    print(f"Extraction complete: {processed} files -> {len(all_chunks_data)} chunks")
    if failed:
        print(f"  {len(failed)} files failed")

    # ── Save BM25 corpus ──
    print(f"\nSaving BM25 corpus ({len(bm25_corpus)} chunks)...")
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(bm25_corpus, f)
    corpus_size_mb = BM25_CORPUS_PATH.stat().st_size / 1024 / 1024
    print(f"  Saved to {BM25_CORPUS_PATH} ({corpus_size_mb:.1f} MB)")

    # ── Embed with Voyage AI ──
    print(f"\nEmbedding {len(all_chunks_data)} chunks with Voyage AI ({VOYAGE_MODEL})...")
    texts_to_embed = [c["text"] for c in all_chunks_data]
    all_embeddings = []

    for batch_start in range(0, len(texts_to_embed), VOYAGE_BATCH_SIZE):
        batch_end = min(batch_start + VOYAGE_BATCH_SIZE, len(texts_to_embed))
        batch = texts_to_embed[batch_start:batch_end]

        for attempt in range(3):
            try:
                result = vo.embed(batch, model=VOYAGE_MODEL, input_type="document")
                all_embeddings.extend(result.embeddings)
                batch_num = batch_start // VOYAGE_BATCH_SIZE + 1
                total_batches = (len(texts_to_embed) + VOYAGE_BATCH_SIZE - 1) // VOYAGE_BATCH_SIZE
                print(f"  Batch {batch_num}/{total_batches} "
                      f"(chunks {batch_start+1}-{batch_end} / {len(texts_to_embed)})")
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                if attempt < 2:
                    print(f"  Rate limited, waiting {wait}s... (attempt {attempt+1}/3)")
                    time.sleep(wait)
                else:
                    print(f"  Failed batch {batch_start}-{batch_end}: {e}")
                    all_embeddings.extend([None] * len(batch))
        time.sleep(1)

    # ── Upsert to Pinecone ──
    print(f"\nUpserting {len(all_chunks_data)} vectors to Pinecone...")
    upserted = 0
    UPSERT_BATCH = 100

    for batch_start in range(0, len(all_chunks_data), UPSERT_BATCH):
        batch_end = min(batch_start + UPSERT_BATCH, len(all_chunks_data))
        vectors = []
        for j in range(batch_start, batch_end):
            if j < len(all_embeddings) and all_embeddings[j] is not None:
                vectors.append({
                    "id": all_chunks_data[j]["id"],
                    "values": all_embeddings[j],
                    "metadata": all_chunks_data[j]["metadata"],
                })
        if vectors:
            try:
                index.upsert(vectors=vectors)
                upserted += len(vectors)
                print(f"  Upserted {upserted} / {len(all_chunks_data)}")
            except Exception as e:
                print(f"  Upsert error at batch {batch_start}: {e}")

    # ── Summary ──
    stats = index.describe_index_stats()
    print(f"\n{'='*70}")
    print(f"INGESTION v2 COMPLETE")
    print(f"  Files processed:     {processed}")
    print(f"  Total chunks:        {len(all_chunks_data)}")
    print(f"  Vectors upserted:    {upserted}")
    print(f"  Pinecone total:      {stats.total_vector_count}")
    print(f"  BM25 corpus:         {BM25_CORPUS_PATH} ({corpus_size_mb:.1f} MB)")
    if failed:
        print(f"\n  Failed files ({len(failed)}):")
        for f in failed[:20]:
            print(f"    - {f}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")
    print(f"{'='*70}")


if __name__ == "__main__":
    run()
