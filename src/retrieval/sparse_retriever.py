import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.config import get_config_value
from src.paths import CHUNKS_DIR
from src.utils.json_utils import load_jsonl


_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}")
_STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "was", "were", "be",
    "being", "been", "the", "a", "an", "of", "to", "in", "on", "for", "with", "by",
    "and", "or", "from", "about", "explain", "define", "describe", "tell", "me", "please",
    "does", "do", "did", "it", "this", "that", "these", "those", "using", "use", "used",
}


class BM25SparseRetriever:
    """Dependency-free BM25 retriever over data/chunks/chunks.jsonl.

    The query side removes common stopwords so generic prompt words such as
    "what is" or "introduction" do not dominate exact keyword retrieval.
    """

    def __init__(
        self,
        chunks_path: str | Path | None = None,
        records: Iterable[Dict[str, Any]] | None = None,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        self.chunks_path = Path(chunks_path or (CHUNKS_DIR / "chunks.jsonl"))
        self.k1 = float(k1 if k1 is not None else get_config_value("settings", "retrieval", "bm25_k1", default=1.5))
        self.b = float(b if b is not None else get_config_value("settings", "retrieval", "bm25_b", default=0.75))
        self.records: List[Dict[str, Any]] = list(records) if records is not None else load_jsonl(self.chunks_path)
        self._doc_tokens: List[List[str]] = []
        self._doc_term_freqs: List[Counter[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_doc_len = 0.0
        self._build_index()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text or "")]

    @classmethod
    def query_tokens(cls, query: str) -> List[str]:
        tokens = cls.tokenize(query)
        filtered = [token for token in tokens if token not in _STOPWORDS]
        return filtered or tokens

    def _record_text_for_sparse(self, record: Dict[str, Any]) -> str:
        metadata_parts = [
            record.get("file_name", ""),
            record.get("document_title", ""),
            record.get("section_name", ""),
        ]
        return " ".join([str(record.get("text", "")), *map(str, metadata_parts)])

    def _build_index(self) -> None:
        doc_freqs: defaultdict[str, int] = defaultdict(int)
        total_len = 0
        self._doc_tokens = []
        self._doc_term_freqs = []

        for record in self.records:
            tokens = self.tokenize(self._record_text_for_sparse(record))
            self._doc_tokens.append(tokens)
            tf = Counter(tokens)
            self._doc_term_freqs.append(tf)
            total_len += len(tokens)
            for token in tf.keys():
                doc_freqs[token] += 1

        self._doc_freqs = dict(doc_freqs)
        self._avg_doc_len = (total_len / len(self.records)) if self.records else 0.0

    def is_ready(self) -> bool:
        return bool(self.records)

    def _idf(self, token: str) -> float:
        n_docs = len(self.records)
        if n_docs == 0:
            return 0.0
        df = self._doc_freqs.get(token, 0)
        return math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))

    def _score_record(self, query_tokens: List[str], index: int) -> float:
        if not query_tokens or not self.records:
            return 0.0
        tf = self._doc_term_freqs[index]
        doc_len = len(self._doc_tokens[index])
        if doc_len == 0 or self._avg_doc_len == 0:
            return 0.0

        score = 0.0
        for token in query_tokens:
            term_freq = tf.get(token, 0)
            if term_freq <= 0:
                continue
            idf = self._idf(token)
            denom = term_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_doc_len))
            score += idf * ((term_freq * (self.k1 + 1.0)) / denom)
        return float(score)

    def retrieve(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if not query.strip() or not self.is_ready():
            return []

        query_tokens = self.query_tokens(query)
        scored: List[tuple[int, float]] = []
        for index, _record in enumerate(self.records):
            score = self._score_record(query_tokens, index)
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for rank, (index, score) in enumerate(scored[: max(1, int(top_k))], start=1):
            record = self.records[index]
            metadata = {
                "doc_id": record.get("doc_id"),
                "chunk_index": record.get("chunk_index"),
                "source_rel": record.get("source_rel"),
                "file_name": record.get("file_name"),
                "file_type": record.get("file_type"),
                "parent_doc_id": record.get("parent_doc_id", record.get("doc_id")),
                "sentence_count": record.get("sentence_count"),
                "start_sentence_index": record.get("start_sentence_index"),
                "end_sentence_index": record.get("end_sentence_index"),
                "document_title": record.get("document_title"),
                "section_name": record.get("section_name"),
                "chunk_position": record.get("chunk_position"),
                "importance_score": record.get("importance_score", 0),
            }
            results.append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "text": record.get("text", ""),
                    "metadata": metadata,
                    "sparse_rank": rank,
                    "sparse_score": round(float(score), 6),
                    "retrieval_method": "sparse",
                }
            )
        return results
