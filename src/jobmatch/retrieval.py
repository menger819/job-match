"""Step 4 / technique 2: retrieval with embeddings (the "R" of RAG), plus BM25.

Narrows hundreds of collected jobs down to ~40 candidates for the (more expensive) LLM reranker:

    hybrid = 0.75 * cosine(embedding(resume), embedding(job)) + 0.25 * normalized BM25

Embeddings capture meaning ("後端工程師" ~ "Backend Engineer"); BM25 rewards exact skill keywords.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Protocol, Sequence

import numpy as np

from jobmatch.models import CandidateProfile, Job, MatchResult

log = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
# bge-m3's main branch only ships pytorch_model.bin, which transformers refuses to load with
# torch < 2.6 (CVE-2025-32434) - and Intel Macs can't get torch >= 2.3. This pinned commit
# (Hugging Face's safetensors conversion) loads safely with any torch version.
BGE_M3_SAFETENSORS_REVISION = "9a0624b896d81da7492a910ffa53731274b6cf3d"
EMBED_WEIGHT = 0.75
BM25_WEIGHT = 0.25


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized vectors, shape (len(texts), dim)."""


class SentenceTransformerEmbedder:
    """Local multilingual embedding model (no API cost, resume never leaves the machine here)."""

    def __init__(self, model_name: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        name = model_name or os.getenv("JOBMATCH_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        log.info("Loading embedding model %s (first run downloads it)", name)
        if name == DEFAULT_EMBED_MODEL:
            self.model = SentenceTransformer(
                name, revision=BGE_M3_SAFETENSORS_REVISION, model_kwargs={"use_safetensors": True}
            )
            return
        try:  # prefer safetensors: works with old torch and is safer to load
            self.model = SentenceTransformer(name, model_kwargs={"use_safetensors": True})
        except OSError:
            self.model = SentenceTransformer(name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True, batch_size=16, show_progress_bar=False)
        )


_CJK = re.compile(r"[㐀-鿿]")
_TOKEN = re.compile(r"[a-z0-9+#.]+|[㐀-鿿]+")


def tokenize(text: str) -> List[str]:
    """Latin words as-is; Chinese runs split into character bigrams (no dictionary needed)."""
    tokens: List[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if _CJK.match(tok):
            tokens.extend(tok[i : i + 2] for i in range(max(1, len(tok) - 1)))
        else:
            tokens.append(tok.strip("."))
    return [t for t in tokens if t]


def dedupe(jobs: List[Job]) -> List[Job]:
    """Phase 1 dedup: same URL, or same company + title (case-insensitive)."""
    seen_urls, seen_pairs, out = set(), set(), []
    for job in jobs:
        pair = (job.company.strip().lower(), job.title.strip().lower())
        if job.url in seen_urls or pair in seen_pairs:
            continue
        seen_urls.add(job.url)
        seen_pairs.add(pair)
        out.append(job)
    return out


def location_matches(job: Job, location: Optional[str]) -> bool:
    if not location:
        return True
    if not job.location:
        return True  # unknown location: keep
    loc, jl = location.lower(), job.location.lower()
    aliases = {
        "taipei": ["taipei", "台北", "臺北"],
        "new taipei": ["new taipei", "新北"],
        "taichung": ["taichung", "台中", "臺中"],
        "tainan": ["tainan", "台南", "臺南"],
        "kaohsiung": ["kaohsiung", "高雄"],
        "hsinchu": ["hsinchu", "新竹"],
        "taoyuan": ["taoyuan", "桃園"],
        "remote": ["remote", "遠端", "在家"],
    }
    names = next((v for k, v in aliases.items() if k in loc or any(a in loc for a in v)), [loc])
    return any(n in jl for n in names) or "remote" in jl or "遠端" in jl


def retrieve(
    profile: CandidateProfile,
    jobs: List[Job],
    embedder: Embedder,
    top_k: int = 40,
    location: Optional[str] = None,
) -> List[MatchResult]:
    if not jobs:
        return []

    # Simple location filter; fall back to all jobs if it removes too many.
    filtered = [j for j in jobs if location_matches(j, location)]
    if len(filtered) < min(top_k, len(jobs)) // 2:
        filtered = jobs

    query_text = profile.to_retrieval_text()
    vecs = embedder.encode([query_text] + [j.to_retrieval_text()[:4000] for j in filtered])
    query_vec, job_vecs = vecs[0], vecs[1:]
    cosine = job_vecs @ query_vec

    bm25 = _bm25_scores(query_text + " " + " ".join(profile.skills), [j.to_retrieval_text() for j in filtered])

    hybrid = EMBED_WEIGHT * cosine + BM25_WEIGHT * bm25
    order = np.argsort(-hybrid)[:top_k]
    return [MatchResult(job=filtered[i], retrieval_score=float(hybrid[i])) for i in order]


def _bm25_scores(query: str, docs: List[str]) -> np.ndarray:
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize(d) or ["_"] for d in docs]
    scores = np.asarray(BM25Okapi(tokenized).get_scores(tokenize(query)), dtype=float)
    top = scores.max() if len(scores) else 0
    return scores / top if top > 0 else np.zeros_like(scores)
