"""104 人力銀行 (www.104.com.tw) collector.

Uses the JSON endpoints that the 104 website itself calls. They need a Referer header.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from jobmatch.collectors.base import Collector
from jobmatch.models import Job

log = logging.getLogger(__name__)

SEARCH_URL = "https://www.104.com.tw/jobs/search/api/jobs"
DETAIL_URL = "https://www.104.com.tw/job/ajax/content/{code}"
REFERER = "https://www.104.com.tw/jobs/search/"

# 104 area codes for common locations (city level).
AREA_CODES = {
    "taipei": "6001001000", "台北": "6001001000", "臺北": "6001001000",
    "new taipei": "6001002000", "新北": "6001002000",
    "keelung": "6001004000", "基隆": "6001004000",
    "taoyuan": "6001005000", "桃園": "6001005000",
    "hsinchu": "6001006000", "新竹": "6001006000",
    "taichung": "6001008000", "台中": "6001008000", "臺中": "6001008000",
    "tainan": "6001014000", "台南": "6001014000", "臺南": "6001014000",
    "kaohsiung": "6001016000", "高雄": "6001016000",
}


def area_code(location: Optional[str]) -> Optional[str]:
    if not location:
        return None
    loc = location.strip().lower()
    for key, code in AREA_CODES.items():
        if key in loc:
            return code
    return None


class Jobs104Collector(Collector):
    name = "104"

    def search(self, query: str, location: Optional[str], limit: int) -> List[Job]:
        jobs: List[Job] = []
        page = 1
        while len(jobs) < limit and page <= 5:
            params = {"keyword": query, "page": page, "pagesize": 20, "order": 15, "jobsource": "m_joblist_search"}
            code = area_code(location)
            if code:
                params["area"] = code
            text = self.client.get_text(SEARCH_URL, params=params, headers={"Referer": REFERER})
            batch = parse_search(text)
            if not batch:
                break
            jobs.extend(batch)
            page += 1
        return jobs[:limit]

    def fetch_detail(self, job: Job) -> Job:
        try:
            text = self.client.get_text(
                DETAIL_URL.format(code=job.job_id),
                headers={"Referer": f"https://www.104.com.tw/job/{job.job_id}"},
            )
            detail = parse_detail(text)
        except Exception as e:  # detail is best-effort; keep the snippet
            log.debug("104 detail failed for %s: %s", job.job_id, e)
            return job
        if detail:
            job.description = detail
        return job


def _job_code(link: str) -> str:
    m = re.search(r"/job/([0-9a-zA-Z]+)", link or "")
    return m.group(1) if m else ""


def parse_search(text: str) -> List[Job]:
    data = json.loads(text)
    items: List[Dict[str, Any]] = []
    if isinstance(data.get("data"), list):  # current API
        items = data["data"]
    elif isinstance(data.get("data"), dict):  # older /jobs/search/list API
        items = data["data"].get("list", [])

    jobs = []
    for it in items:
        link = (it.get("link") or {}).get("job", "") if isinstance(it.get("link"), dict) else ""
        code = _job_code(link) or str(it.get("jobNo") or "")
        title = (it.get("jobName") or "").strip()
        company = (it.get("custName") or "").strip()
        if not (code and title):
            continue
        salary = ""
        low, high = it.get("salaryLow"), it.get("salaryHigh")
        if low or high:
            salary = f"{low or ''}~{high or ''}"
        jobs.append(
            Job(
                platform="104",
                job_id=code,
                title=title,
                company=company,
                url=f"https://www.104.com.tw/job/{code}",
                location=it.get("jobAddrNoDesc") or it.get("jobAddress") or "",
                description=(it.get("description") or it.get("descSnippet") or "").strip(),
                salary=salary,
                posted_at=str(it.get("appearDate") or ""),
            )
        )
    return jobs


def parse_detail(text: str) -> str:
    data = json.loads(text).get("data") or {}
    job = data.get("jobDetail") or {}
    cond = data.get("condition") or {}
    parts = [job.get("jobDescription", "")]
    skills = [s.get("description", "") for s in cond.get("specialty", []) if isinstance(s, dict)]
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    if cond.get("other"):
        parts.append(cond["other"])
    return "\n".join(p for p in parts if p).strip()
