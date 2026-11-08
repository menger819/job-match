"""Data models shared across the pipeline."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Structured view of a resume, produced by the LLM (technique 1)."""

    headline: str = Field(description="One-line professional headline")
    summary: str = Field(description="2-3 sentence summary of the candidate")
    years_experience: float = 0
    seniority: str = Field(description="intern | junior | mid | senior | lead | manager | director")
    job_titles: List[str] = Field(default_factory=list, description="Titles held or targeted")
    skills: List[str] = Field(default_factory=list)
    industries: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    preferred_locations: List[str] = Field(default_factory=list)
    search_queries_zh: List[str] = Field(default_factory=list, description="Short job-search keywords in Traditional Chinese")
    search_queries_en: List[str] = Field(default_factory=list, description="Short job-search keywords in English")

    def to_retrieval_text(self) -> str:
        """Text used as the 'query' side of embedding retrieval."""
        parts = [
            self.headline,
            self.summary,
            "Titles: " + ", ".join(self.job_titles),
            "Skills: " + ", ".join(self.skills),
            "Industries: " + ", ".join(self.industries),
            "Seniority: " + self.seniority,
        ]
        return "\n".join(p for p in parts if p)


class Job(BaseModel):
    """A job posting normalized across platforms."""

    platform: str  # "104" | "cake" | "linkedin"
    job_id: str
    title: str
    company: str
    url: str
    location: str = ""
    description: str = ""
    salary: str = ""
    posted_at: str = ""

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.job_id}"

    def to_retrieval_text(self) -> str:
        return f"{self.title}\n{self.company}\n{self.location}\n{self.description}".strip()


class MatchResult(BaseModel):
    job: Job
    retrieval_score: float = 0.0
    llm_score: Optional[float] = None
    reason: str = ""

    @property
    def final_score(self) -> float:
        return self.llm_score if self.llm_score is not None else self.retrieval_score * 100
