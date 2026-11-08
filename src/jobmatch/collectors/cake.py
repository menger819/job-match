"""Cake (www.cake.me) collector.

Cake's job pages are server-rendered (Next.js). We read job links from the search page HTML
and the embedded __NEXT_DATA__ JSON, and the description from each job page.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from jobmatch.collectors.base import Collector
from jobmatch.models import Job

log = logging.getLogger(__name__)

BASE = "https://www.cake.me"
JOB_PATH = re.compile(r"^/companies/([^/?#]+)/jobs/([^/?#]+)")


class CakeCollector(Collector):
    name = "cake"

    def search(self, query: str, location: Optional[str], limit: int) -> List[Job]:
        try:
            return self._pages(f"{BASE}/jobs/{quote(query)}", location, limit)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 403:
                raise
            # Cake puts keyword (and location) search behind a bot challenge. We don't try to
            # get around it: fall back to the public, unfiltered job listing, and let the
            # retrieval step filter by location and rank by relevance.
            log.info("cake keyword search refused (403); browsing the job listing instead")
            return self._pages(f"{BASE}/jobs", None, limit, max_pages=6)

    def _pages(self, url: str, location: Optional[str], limit: int, max_pages: int = 3) -> List[Job]:
        jobs: List[Job] = []
        seen = set()
        for page in range(1, max_pages + 1):
            params: Dict[str, Any] = {"page": page}
            if location and location.lower() != "remote":
                params["location_list[0]"] = location
            html = self.client.get_text(url, params=params)
            batch = [j for j in parse_search(html) if j.key not in seen]
            if not batch:
                break
            for j in batch:
                seen.add(j.key)
            jobs.extend(batch)
            if len(jobs) >= limit:
                break
        return jobs[:limit]

    def fetch_detail(self, job: Job) -> Job:
        try:
            html = self.client.get_text(job.url)
            detail = parse_detail(html)
        except Exception as e:
            log.debug("cake detail failed for %s: %s", job.url, e)
            return job
        if detail:
            job.description = detail
        return job


def _walk(obj: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _from_next_data(soup: BeautifulSoup) -> List[Job]:
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return []
    try:
        data = json.loads(tag.string)
    except json.JSONDecodeError:
        return []
    jobs = []
    for node in _walk(data):
        page = node.get("page")
        if not (isinstance(page, dict) and node.get("title") and node.get("path") and page.get("path")):
            continue
        locations = node.get("locations") or node.get("location_list") or []
        jobs.append(
            Job(
                platform="cake",
                job_id=f"{page['path']}/{node['path']}",
                title=str(node["title"]).strip(),
                company=str(page.get("name") or page["path"]).strip(),
                url=f"{BASE}/companies/{page['path']}/jobs/{node['path']}",
                location=", ".join(str(x) for x in locations) if isinstance(locations, list) else str(locations),
                description=str(node.get("description_plain_text") or node.get("description") or "")[:2000],
            )
        )
    return jobs


def _from_links(soup: BeautifulSoup) -> List[Job]:
    jobs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].replace(BASE, "")
        m = JOB_PATH.match(href)
        title = a.get_text(" ", strip=True)
        if not m or not title:
            continue
        company_slug, job_slug = m.group(1), m.group(2)
        company = company_slug
        card = a.find_parent(["div", "li", "article"])
        if card:
            comp_a = card.find("a", href=re.compile(rf"/companies/{re.escape(company_slug)}/?$"))
            if comp_a and comp_a.get_text(strip=True):
                company = comp_a.get_text(strip=True)
        jobs.append(
            Job(
                platform="cake",
                job_id=f"{company_slug}/{job_slug}",
                title=title,
                company=company,
                url=f"{BASE}/companies/{company_slug}/jobs/{job_slug}",
            )
        )
    return jobs


def parse_search(html: str) -> List[Job]:
    soup = BeautifulSoup(html, "html.parser")
    jobs = _from_next_data(soup) or _from_links(soup)
    unique: Dict[str, Job] = {}
    for j in jobs:
        unique.setdefault(j.key, j)
    return list(unique.values())


def parse_detail(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for node in _walk(_next_data(soup)):
        for key in ("description_plain_text", "description"):
            val = node.get(key)
            if isinstance(val, str) and len(val) > 200:
                return BeautifulSoup(val, "html.parser").get_text("\n", strip=True)
    meta = soup.find("meta", attrs={"name": "description"})
    return (meta.get("content") or "").strip() if meta else ""


def _next_data(soup: BeautifulSoup) -> Any:
    tag = soup.find("script", id="__NEXT_DATA__")
    try:
        return json.loads(tag.string) if tag and tag.string else {}
    except json.JSONDecodeError:
        return {}
