"""LinkedIn collector (optional).

LinkedIn restricts automated access, and its User Agreement prohibits scraping. This collector
uses only the public, logged-out "guest" job listing endpoints, at a low request rate, and can
be turned off with LINKEDIN_ENABLED=false or --platforms 104,cake. If it gets blocked
(HTTP 429/999) the pipeline continues with the other platforms.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from bs4 import BeautifulSoup

from jobmatch.collectors.base import Collector
from jobmatch.models import Job

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
DETAIL_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"


class LinkedInCollector(Collector):
    name = "linkedin"

    def search(self, query: str, location: Optional[str], limit: int) -> List[Job]:
        jobs: List[Job] = []
        start = 0
        while len(jobs) < limit and start < 100:
            params = {"keywords": query, "location": location or "Taiwan", "start": start}
            html = self.client.get_text(SEARCH_URL, params=params)
            batch = parse_search(html)
            if not batch:
                break
            jobs.extend(batch)
            start += 25
        return jobs[:limit]

    def fetch_detail(self, job: Job) -> Job:
        try:
            html = self.client.get_text(DETAIL_URL.format(job_id=job.job_id))
            detail = parse_detail(html)
        except Exception as e:
            log.debug("linkedin detail failed for %s: %s", job.job_id, e)
            return job
        if detail:
            job.description = detail
        return job


def _text(node) -> str:
    return node.get_text(" ", strip=True) if node else ""


def parse_search(html: str) -> List[Job]:
    soup = BeautifulSoup(html, "html.parser")
    jobs = []
    for card in soup.select("div.base-card, li"):
        link = card.select_one("a.base-card__full-link") or card.find("a", href=re.compile(r"/jobs/view/"))
        if not link:
            continue
        href = link["href"].split("?")[0]
        m = re.search(r"-(\d+)/?$", href) or re.search(r"/jobs/view/(\d+)", href)
        urn = card.get("data-entity-urn", "")
        job_id = urn.rsplit(":", 1)[-1] if urn else (m.group(1) if m else "")
        title = _text(card.select_one(".base-search-card__title")) or _text(link)
        company = _text(card.select_one(".base-search-card__subtitle"))
        if not (job_id and title):
            continue
        time_tag = card.find("time")
        jobs.append(
            Job(
                platform="linkedin",
                job_id=job_id,
                title=title,
                company=company,
                url=f"https://www.linkedin.com/jobs/view/{job_id}",
                location=_text(card.select_one(".job-search-card__location")),
                posted_at=time_tag.get("datetime", "") if time_tag else "",
            )
        )
    unique = {}
    for j in jobs:
        unique.setdefault(j.key, j)
    return list(unique.values())


def parse_detail(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one(".show-more-less-html__markup") or soup.select_one(".description__text")
    return node.get_text("\n", strip=True) if node else ""
