"""Tiny, inspectable TF-IDF retrieval pipeline for the course scaffold."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# The live application currently specializes in debt management and student loans.
# Keep earlier investment-course material in the repository, but out of this index.
STUDENT_LOAN_CORPUS_PATTERNS = (
    "07_student_loan_*.md",
    "08_public_service_loan_forgiveness_*.md",
    "09_student_loan_tax_rules_*.md",
    "10_household_finances_shed_*.md",
)

@dataclass
class Chunk:
    source: str
    chunk_id: str
    text: str

@dataclass
class Hit:
    source: str
    chunk_id: str
    text: str
    score: float

    def to_dict(self):
        return asdict(self)


def _is_provenance_only(paragraph: str) -> bool:
    """Exclude source labels/URL lists that are useful to readers but not retrieval evidence."""
    normalized = paragraph.strip().lower()
    return (
        normalized in {"source:", "sources:"}
        or normalized.startswith("source: ")
        or normalized.startswith("sources: ")
        or normalized.startswith("- internal revenue service,")
    )


def load_chunks(
    corpus_dir: str | Path,
    include_patterns: tuple[str, ...] | None = None,
) -> List[Chunk]:
    corpus_dir = Path(corpus_dir)
    chunks: List[Chunk] = []
    if include_patterns is None:
        paths = sorted(corpus_dir.glob("*.md"))
    else:
        paths = sorted({path for pattern in include_patterns for path in corpus_dir.glob(pattern)})

    for path in paths:
        body = path.read_text(encoding="utf-8")
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", body)]
        paragraphs = [
            p
            for p in paragraphs
            if p and not p.startswith("#") and not _is_provenance_only(p)
        ]
        for i, paragraph in enumerate(paragraphs, 1):
            chunks.append(Chunk(path.name, f"{path.stem}-c{i}", paragraph))
    return chunks


class Retriever:
    def __init__(
        self,
        corpus_dir: str | Path,
        include_patterns: tuple[str, ...] | None = None,
    ):
        self.chunks = load_chunks(corpus_dir, include_patterns)
        if not self.chunks:
            raise ValueError("No corpus chunks found")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> List[Hit]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        order = [i for i in scores.argsort()[::-1] if scores[i] >= min_score][:top_k]
        return [
            Hit(
                source=self.chunks[i].source,
                chunk_id=self.chunks[i].chunk_id,
                text=self.chunks[i].text,
                score=float(scores[i]),
            )
            for i in order
        ]


def format_context(hits: List[Hit]) -> str:
    sections = []
    for hit in hits:
        sections.append(f"SOURCE [{hit.chunk_id}] ({hit.source})\n{hit.text}")
    return "\n\n".join(sections)
