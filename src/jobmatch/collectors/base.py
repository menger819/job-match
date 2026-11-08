"""Shared plumbing for platform collectors: polite HTTP, retries, on-disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from jobmatch.models import Job

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


class BlockedError(Exception):
    """The platform refused our requests (rate limit / bot protection)."""


class PoliteClient:
    """httpx wrapper: minimum delay between requests, retry with backoff, 24h file cache."""

    def __init__(
        self,
        cache_dir: Path = Path(".cache/http"),
        use_cache: bool = True,
        min_delay: float = 1.0,
        ttl_seconds: int = 24 * 3600,
        timeout: float = 20.0,
    ):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.min_delay = min_delay
        self.ttl = ttl_seconds
        self._last = 0.0
        self._http = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8"},
        )

    def get_text(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> str:
        full = str(httpx.URL(url, params=params or {}))
        cache_file = self.cache_dir / (hashlib.sha256(full.encode()).hexdigest()[:32] + ".json")

        if self.use_cache and cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if time.time() - cached["ts"] < self.ttl:
                return cached["text"]

        text = self._fetch(full, headers or {})
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"ts": time.time(), "url": full, "text": text}), encoding="utf-8")
        return text

    def _fetch(self, url: str, headers: Dict, attempts: int = 3) -> str:
        for attempt in range(attempts):
            wait = self.min_delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait + random.uniform(0, 0.5))
            self._last = time.time()
            try:
                resp = self._http.get(url, headers=headers)
            except httpx.HTTPError as e:
                log.debug("network error %s on %s", e, url)
                if attempt == attempts - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (429, 999) or resp.status_code >= 500:
                log.debug("HTTP %s on %s (attempt %d)", resp.status_code, url, attempt + 1)
                if attempt == attempts - 1:
                    if resp.status_code in (429, 999):
                        raise BlockedError(f"HTTP {resp.status_code}")
                    resp.raise_for_status()
                time.sleep(3 * 2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.text
        raise RuntimeError("unreachable")


class Collector(ABC):
    """One job platform. Subclasses implement search() and optionally fetch_detail()."""

    name: str = "base"

    def __init__(self, client: PoliteClient):
        self.client = client

    @abstractmethod
    def search(self, query: str, location: Optional[str], limit: int) -> List[Job]:
        """Return up to `limit` jobs for a keyword query."""

    def fetch_detail(self, job: Job) -> Job:
        """Fill in the full job description. Default: return the job unchanged."""
        return job
