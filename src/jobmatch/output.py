"""Step 6: format and save the top-N list as `<Company> - [<Job Title>](<URL>)`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from jobmatch.html_report import to_html
from jobmatch.models import CandidateProfile, MatchResult

PLATFORM_LABEL = {"104": "104", "cake": "Cake", "linkedin": "LinkedIn"}


def format_line(rank: int, m: MatchResult) -> str:
    company = m.job.company or "(unknown company)"
    return f"{rank}. {company} - [{m.job.title}]({m.job.url})"


def to_terminal(results: List[MatchResult]) -> str:
    lines = []
    for i, m in enumerate(results, 1):
        lines.append(f"{format_line(i, m)}  ⭐ {m.final_score:.0f}  ({PLATFORM_LABEL.get(m.job.platform, m.job.platform)})")
        if m.reason:
            lines.append(f"     {m.reason}")
    return "\n".join(lines)


def to_markdown(results: List[MatchResult], profile: CandidateProfile) -> str:
    out = [
        "# Top Job Matches",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} for: {profile.headline}_",
        "",
    ]
    for i, m in enumerate(results, 1):
        out.append(format_line(i, m))
    out += ["", "## Details", "", "| # | Score | Platform | Location | Why |", "|---|---|---|---|---|"]
    for i, m in enumerate(results, 1):
        why = (m.reason or "").replace("|", "/")
        out.append(f"| {i} | {m.final_score:.0f} | {PLATFORM_LABEL.get(m.job.platform, m.job.platform)} | {m.job.location} | {why} |")
    return "\n".join(out) + "\n"


def save(results: List[MatchResult], profile: CandidateProfile, output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md = output_dir / "results.md"
    js = output_dir / "results.json"
    md.write_text(to_markdown(results, profile), encoding="utf-8")
    payload = {
        "profile": profile.model_dump(),
        "results": [
            {
                "rank": i,
                "company": m.job.company,
                "title": m.job.title,
                "url": m.job.url,
                "platform": m.job.platform,
                "location": m.job.location,
                "salary": m.job.salary,
                "llm_score": m.llm_score,
                "retrieval_score": round(m.retrieval_score, 4),
                "reason": m.reason,
            }
            for i, m in enumerate(results, 1)
        ],
    }
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html = output_dir / "results.html"
    html.write_text(to_html(results, profile), encoding="utf-8")
    return [md, js, html]
