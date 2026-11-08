from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import numpy as np
import pytest

from jobmatch.collectors.base import Collector
from jobmatch.models import Job
from jobmatch.retrieval import tokenize

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Hashing bag-of-words embedder: deterministic, no model download."""

    dim = 256

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


class FakeLLM:
    """Returns canned profile JSON, and scores jobs by whether 'python' is in the job text."""

    def __init__(self, profile: dict):
        self.profile = profile
        self.calls: List[dict] = []

    def complete_json(self, system, content, schema, max_tokens=16000):
        self.calls.append({"system": system, "content": content, "schema": schema})
        if "scores" in schema["properties"]:
            text = content[-1]["text"]
            scores = []
            for block in text.split("<job ")[1:]:
                job_id = block.split('job_id="', 1)[1].split('"', 1)[0]
                score = 90 if "python" in block.lower() else 30
                scores.append({"job_id": job_id, "score": score, "reason": "fake reason"})
            return {"scores": scores}
        return dict(self.profile)


class FakeCollector(Collector):
    def __init__(self, name: str, jobs: List[Job]):
        self.name = name
        self.jobs = jobs
        self.detail_calls = 0

    def search(self, query, location, limit):
        return self.jobs[:limit]

    def fetch_detail(self, job):
        self.detail_calls += 1
        return job


PROFILE = {
    "headline": "Backend engineer, Python",
    "summary": "Backend engineer with 5 years of Python and AWS.",
    "years_experience": 5,
    "seniority": "senior",
    "job_titles": ["Backend Engineer"],
    "skills": ["Python", "Django", "AWS", "PostgreSQL"],
    "industries": ["E-commerce"],
    "languages": ["Chinese", "English"],
    "preferred_locations": ["Taipei"],
    "search_queries_zh": ["後端工程師 Python"],
    "search_queries_en": ["Backend Engineer Python"],
}


@pytest.fixture
def profile_dict():
    return dict(PROFILE)


@pytest.fixture
def fixtures_dir():
    return FIXTURES


def make_jobs(platform: str, n: int, python_every: int = 2) -> List[Job]:
    jobs = []
    for i in range(n):
        has_py = i % python_every == 0
        jobs.append(
            Job(
                platform=platform,
                job_id=f"{platform}-{i}",
                title=("Python Backend Engineer" if has_py else "Sales Manager") + f" {i}",
                company=f"Company {platform} {i}",
                url=f"https://example.com/{platform}/{i}",
                location="台北市信義區",
                description=("Python Django AWS PostgreSQL backend APIs" if has_py else "B2B sales quota CRM"),
            )
        )
    return jobs
