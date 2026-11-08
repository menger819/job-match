"""Step 2 / technique 1: LLM structured extraction.

Claude reads the raw resume and returns a CandidateProfile (skills, titles, seniority...)
plus bilingual search keywords: 104 and Cake are mostly Chinese, LinkedIn mostly English.
"""

from __future__ import annotations

from typing import Optional

from jobmatch.llm import JSONLLM, strict_object
from jobmatch.models import CandidateProfile

_STR_LIST = {"type": "array", "items": {"type": "string"}}

PROFILE_SCHEMA = strict_object(
    {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "years_experience": {"type": "number"},
        "seniority": {
            "type": "string",
            "enum": ["intern", "junior", "mid", "senior", "lead", "manager", "director"],
        },
        "job_titles": _STR_LIST,
        "skills": _STR_LIST,
        "industries": _STR_LIST,
        "languages": _STR_LIST,
        "preferred_locations": _STR_LIST,
        "search_queries_zh": _STR_LIST,
        "search_queries_en": _STR_LIST,
    }
)

SYSTEM = """You are an expert technical recruiter in Taiwan. You read a resume and produce a
structured candidate profile used to search job boards (104.com.tw, cake.me, linkedin.com).

Guidelines:
- skills: concrete, searchable skills and tools (e.g. "Python", "Kubernetes", "財務分析"), most important first, at most 25.
- job_titles: titles the person has held or is clearly qualified for next, most relevant first, at most 6.
- search_queries_zh: 3-5 short keyword queries in Traditional Chinese as a Taiwanese job seeker would type
  them into 104 (e.g. "後端工程師 Python", "資料工程師"). English tech terms may stay in English.
- search_queries_en: 3-5 short English queries for LinkedIn (e.g. "Senior Backend Engineer").
- preferred_locations: only if stated or strongly implied by the resume; otherwise empty.
- Base everything on the resume. Do not invent experience."""


def extract_profile(resume_text: str, llm: JSONLLM, location: Optional[str] = None) -> CandidateProfile:
    content = [
        {
            "type": "text",
            "text": f"<resume>\n{resume_text}\n</resume>",
        },
        {
            "type": "text",
            "text": "Extract the candidate profile."
            + (f" The candidate wants jobs in: {location}." if location else ""),
        },
    ]
    data = llm.complete_json(SYSTEM, content, PROFILE_SCHEMA, max_tokens=8000)
    profile = CandidateProfile.model_validate(data)
    if location and location not in profile.preferred_locations:
        profile.preferred_locations.insert(0, location)
    return profile
