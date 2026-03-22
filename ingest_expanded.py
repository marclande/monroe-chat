"""
Monroe Institute — Ingest Expanded Archives
=============================================
Processes the newly downloaded RAM (Robert A. Monroe) transcripts
and adds them to the existing Pinecone index.
"""

import os
import re
import sys
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
from pinecone import Pinecone
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────────

EXPANDED_DIR = Path("C:/Users/marc.lande/Downloads/monroe_archive/monroe_archive/transcripts_expanded")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "monroe-explorer")
CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
VOYAGE_MODEL = "voyage-3"
VOYAGE_BATCH_SIZE = 100
EMBEDDING_DIM = 1024

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


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


def chunk_text(text: str, target_tokens: int = CHUNK_TARGET_TOKENS,
               overlap_tokens: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    if not text.strip():
        return []

    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens > target_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > target_tokens and current_chunk:
                    chunk_text_str = "\n\n".join(current_chunk)
                    chunks.append(chunk_text_str)
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

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def make_chunk_id(category: str, filename: str, chunk_index: int) -> str:
    raw = f"{category}:{filename}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def run():
    print("=" * 70)
    print("Monroe Institute — Expanded Archives Ingestion")
    print("=" * 70)

    # Initialize services
    print("\nConnecting to Voyage AI...")
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX_NAME)

    stats = index.describe_index_stats()
    print(f"Current vectors in index: {stats.total_vector_count}")

    # Discover all transcript files in expanded dir
    print(f"\nScanning {EXPANDED_DIR} for transcripts...")

    all_files = []
    for category_dir in sorted(EXPANDED_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for f in sorted(category_dir.iterdir()):
            # Only process PDF transcripts and _djvu.txt files
            # Skip outline PDFs (we want full transcripts), but include outline _djvu.txt
            if f.suffix.lower() == ".pdf" and "(Transcript)" in f.name:
                all_files.append((category, f))
            elif f.name.endswith("_djvu.txt") and "(Transcript)" in f.name:
                # Only use djvu.txt if we don't have the PDF version
                pdf_name = f.name.replace("_djvu.txt", ".pdf")
                pdf_path = f.parent / pdf_name
                if not pdf_path.exists():
                    all_files.append((category, f))

    print(f"Found {len(all_files)} transcript files to process")

    # Process files
    print("\nProcessing transcripts...\n")
    all_chunks_data = []
    total_files = 0
    failed = []

    for i, (category, filepath) in enumerate(all_files):
        print(f"  [{i + 1}/{len(all_files)}] {category}/{filepath.name}...", end=" ")

        if filepath.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)

        if not text or len(text.strip()) < 50:
            print("No text extracted")
            failed.append(str(filepath))
            continue

        chunks = chunk_text(text)
        if not chunks:
            print("No chunks")
            failed.append(str(filepath))
            continue

        # Derive metadata from filename
        # Category gives us the section (RAM_Interviews, RAM_Talks, etc.)
        explorer_id = "RAM"  # Robert A. Monroe
        if "INSCOM" in category:
            explorer_id = "RAM_INSCOM"
        elif "Guidelines" in category:
            explorer_id = "RAM_Guidelines"
        elif "Gateway" in category:
            explorer_id = "RAM_Gateway"

        for ci, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(category, filepath.name, ci)
            all_chunks_data.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": {
                    "explorer_id": explorer_id,
                    "session_number": 0,
                    "filename": filepath.name,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                    "topics": f"Robert Monroe, {category.replace('_', ' ')}",
                    "text": chunk[:1000],
                },
            })

        total_files += 1
        print(f"{len(chunks)} chunks ({count_tokens(text)} tokens)")

    print(f"\nExtraction complete: {total_files} files -> {len(all_chunks_data)} chunks")
    if failed:
        print(f"{len(failed)} files failed")

    if not all_chunks_data:
        print("No chunks to embed. Done.")
        return

    # Embed with Voyage AI
    print(f"\nEmbedding {len(all_chunks_data)} chunks with Voyage AI...")
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
                print(f"  Batch {batch_num}/{total_batches} ({batch_start + 1}-{batch_end})")
                break
            except Exception as e:
                wait = 22 * (attempt + 1)
                if attempt < 2:
                    print(f"  Rate limited, waiting {wait}s... (attempt {attempt + 1}/3)")
                    time.sleep(wait)
                else:
                    print(f"  Failed batch {batch_start}-{batch_end}: {e}")
                    all_embeddings.extend([None] * len(batch))

        time.sleep(1)  # Small delay between batches

    # Upsert to Pinecone
    print(f"\nUpserting {len(all_chunks_data)} vectors to Pinecone...")
    upserted = 0

    for batch_start in range(0, len(all_chunks_data), 100):
        batch_end = min(batch_start + 100, len(all_chunks_data))
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
                print(f"  Upsert error: {e}")

    # Final stats
    stats = index.describe_index_stats()
    print(f"\n{'=' * 70}")
    print(f"DONE!")
    print(f"  New files processed: {total_files}")
    print(f"  New chunks embedded: {len(all_chunks_data)}")
    print(f"  Vectors upserted: {upserted}")
    print(f"  Total vectors in Pinecone: {stats.total_vector_count}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
