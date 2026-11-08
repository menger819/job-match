"""Glue: resume -> profile -> collect -> retrieve -> enrich -> rerank -> top N."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from jobmatch.collectors import REGISTRY, BlockedError, Collector, PoliteClient
from jobmatch.llm import JSONLLM
from jobmatch.models import CandidateProfile, Job, MatchResult
from jobmatch.profiler import extract_profile
from jobmatch.reranker import rerank
from jobmatch.resume_loader import load_resume
from jobmatch.retrieval import Embedder, dedupe, retrieve

log = logging.getLogger(__name__)


@dataclass
class Settings:
    location: Optional[str] = None
    top: int = 10
    platforms: List[str] = field(default_factory=lambda: ["104", "cake", "linkedin"])
    max_jobs_per_platform: int = 60
    candidates: int = 40  # how many retrieved jobs go to the LLM reranker
    use_cache: bool = True
    cache_dir: Path = Path(".cache/http")


@dataclass
class RunResult:
    profile: CandidateProfile
    results: List[MatchResult]
    collected: Dict[str, int]
    errors: Dict[str, str]


def queries_for(platform: str, profile: CandidateProfile) -> List[str]:
    zh, en = profile.search_queries_zh, profile.search_queries_en
    if platform == "linkedin":
        qs = en[:3] or profile.job_titles[:3]
    else:  # 104 and Cake: mostly Chinese postings, but English tech titles also work
        qs = zh[:2] + en[:1]
    return [q for q in dict.fromkeys(qs) if q.strip()] or profile.job_titles[:2]


def collect(
    profile: CandidateProfile,
    settings: Settings,
    collectors: Dict[str, Collector],
) -> tuple:
    per_query = max(10, settings.max_jobs_per_platform // 3)

    def run(name: str):
        coll = collectors[name]
        jobs: List[Job] = []
        for q in queries_for(name, profile):
            try:
                jobs.extend(coll.search(q, settings.location, per_query))
            except BlockedError as e:
                return name, jobs, f"blocked ({e})"
            except Exception as e:  # one platform failing must not stop the run
                log.debug("collector %s failed", name, exc_info=True)
                return name, jobs, f"{type(e).__name__}: {e}"
            if len(jobs) >= settings.max_jobs_per_platform:
                break
        return name, jobs[: settings.max_jobs_per_platform], ""

    all_jobs: List[Job] = []
    counts: Dict[str, int] = {}
    errors: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(collectors) or 1) as pool:
        for name, jobs, err in pool.map(run, list(collectors)):
            counts[name] = len(jobs)
            all_jobs.extend(jobs)
            if err:
                errors[name] = err
    return all_jobs, counts, errors


def enrich(candidates: List[MatchResult], collectors: Dict[str, Collector]) -> None:
    """Fetch full descriptions only for the shortlisted jobs (keeps request count low)."""

    def run(name: str):
        for m in candidates:
            if m.job.platform == name and len(m.job.description) < 300:
                collectors[name].fetch_detail(m.job)

    with ThreadPoolExecutor(max_workers=len(collectors) or 1) as pool:
        list(pool.map(run, list(collectors)))


def build_collectors(settings: Settings) -> Dict[str, Collector]:
    out = {}
    for name in settings.platforms:
        cls = REGISTRY[name]
        delay = 2.5 if name == "linkedin" else 1.0
        client = PoliteClient(cache_dir=settings.cache_dir / name, use_cache=settings.use_cache, min_delay=delay)
        out[name] = cls(client)
    return out


def run_pipeline(
    resume_path: Path,
    settings: Settings,
    llm: JSONLLM,
    embedder_factory: Callable[[], Embedder],
    collectors: Optional[Dict[str, Collector]] = None,
    progress: Callable[[str], None] = print,
) -> RunResult:
    progress("[1/5] Reading resume ...")
    resume_text = load_resume(resume_path)
    # Load the local matching model first, so setup problems show up before any AI usage is spent.
    progress("      loading matching model (first run downloads ~2 GB) ...")
    embedder = embedder_factory()

    progress("[2/5] Analyzing resume with AI ...")
    profile = extract_profile(resume_text, llm, settings.location)
    progress(f"      skills: {', '.join(profile.skills[:8])}")
    progress(f"      queries: {', '.join(profile.search_queries_zh[:3] + profile.search_queries_en[:3])}")

    collectors = collectors if collectors is not None else build_collectors(settings)
    progress(f"[3/5] Searching {' / '.join(collectors)} ...")
    jobs, counts, errors = collect(profile, settings, collectors)
    jobs = dedupe(jobs)
    summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
    progress(f"      {len(jobs)} unique jobs found ({summary})")
    for name, err in errors.items():
        progress(f"      ⚠️  {name}: {err} - continuing with the other platforms")
    if not jobs:
        return RunResult(profile, [], counts, errors)

    progress("[4/5] Matching (embeddings + BM25) ...")
    candidates = retrieve(profile, jobs, embedder, top_k=settings.candidates, location=settings.location)
    enrich(candidates, collectors)

    progress(f"[5/5] AI reranking top {len(candidates)} ...")
    ranked = rerank(profile, resume_text, candidates, llm)
    return RunResult(profile, ranked[: settings.top], counts, errors)
