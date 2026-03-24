"""
BM25 Keyword Search Index
==========================
Loads the chunk corpus and provides BM25 keyword search
for hybrid retrieval alongside vector search.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


class BM25Index:
    def __init__(self, corpus_path: str = "bm25_corpus.json"):
        self.corpus_path = Path(corpus_path)
        self.chunk_ids = []
        self.chunk_texts = []
        self.bm25 = None
        self._load()

    def _load(self):
        """Load corpus and build BM25 index."""
        if not self.corpus_path.exists():
            print(f"BM25 corpus not found at {self.corpus_path}")
            return

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        self.chunk_ids = list(corpus.keys())
        self.chunk_texts = list(corpus.values())

        # Tokenize all documents
        tokenized = [tokenize(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built: {len(self.chunk_ids)} documents")

    def search(self, query: str, top_k: int = 25) -> list[tuple[str, float]]:
        """Search BM25 index, return list of (chunk_id, score)."""
        if self.bm25 is None:
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top_k indices sorted by score
        scored = [(self.chunk_ids[i], float(scores[i])) for i in range(len(scores))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None
