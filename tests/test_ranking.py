from conftest import FakeCollector, FakeEmbedder, FakeLLM, make_jobs

from jobmatch.models import CandidateProfile, Job
from jobmatch.output import format_line, to_markdown
from jobmatch.pipeline import Settings, queries_for, run_pipeline
from jobmatch.profiler import extract_profile
from jobmatch.reranker import rerank
from jobmatch.retrieval import dedupe, location_matches, retrieve, tokenize


def test_tokenize_mixed_language():
    toks = tokenize("Python 後端工程師 C++")
    assert "python" in toks and "c++" in toks
    assert "後端" in toks and "工程" in toks


def test_dedupe_by_url_and_company_title():
    a = Job(platform="104", job_id="1", title="Backend Engineer", company="Acme", url="u1")
    b = Job(platform="cake", job_id="2", title="backend engineer ", company="ACME", url="u2")
    c = Job(platform="linkedin", job_id="3", title="Other", company="Acme", url="u1")
    d = Job(platform="104", job_id="4", title="Data Engineer", company="Acme", url="u4")
    assert [j.job_id for j in dedupe([a, b, c, d])] == ["1", "4"]


def test_location_matches():
    j = Job(platform="104", job_id="1", title="t", company="c", url="u", location="台北市信義區")
    assert location_matches(j, "Taipei")
    assert not location_matches(j, "Kaohsiung")
    assert location_matches(j, None)
    assert location_matches(Job(platform="x", job_id="2", title="t", company="c", url="u"), "Taipei")


def test_retrieve_ranks_relevant_jobs_first(profile_dict):
    profile = CandidateProfile.model_validate(profile_dict)
    jobs = make_jobs("104", 20)
    top = retrieve(profile, jobs, FakeEmbedder(), top_k=5)
    assert len(top) == 5
    assert all("Python" in m.job.title for m in top)


def test_rerank_uses_llm_scores_and_caches_resume(profile_dict):
    profile = CandidateProfile.model_validate(profile_dict)
    jobs = make_jobs("104", 25, python_every=5)
    candidates = retrieve(profile, jobs, FakeEmbedder(), top_k=25)
    llm = FakeLLM(profile_dict)
    ranked = rerank(profile, "resume text", candidates, llm)
    assert len(llm.calls) == 3  # 25 jobs / batch of 10
    assert all(call["content"][0].get("cache_control") for call in llm.calls)
    assert ranked[0].llm_score == 90
    assert all("Python" in m.job.title for m in ranked[:5])


def test_extract_profile_adds_location(profile_dict):
    profile = extract_profile("resume", FakeLLM(profile_dict), location="Taichung")
    assert profile.preferred_locations[0] == "Taichung"


def test_queries_for_platforms(profile_dict):
    profile = CandidateProfile.model_validate(profile_dict)
    assert queries_for("linkedin", profile) == ["Backend Engineer Python"]
    assert queries_for("104", profile) == ["後端工程師 Python", "Backend Engineer Python"]


def test_output_format():
    from jobmatch.models import MatchResult

    m = MatchResult(job=Job(platform="104", job_id="1", title="Backend Engineer", company="Acme", url="https://x/1"))
    assert format_line(3, m) == "3. Acme - [Backend Engineer](https://x/1)"


def test_pipeline_end_to_end(tmp_path, profile_dict):
    resume = tmp_path / "resume.md"
    resume.write_text("# Jane\nSenior backend engineer. Python, Django, AWS, PostgreSQL. 5 years.", encoding="utf-8")

    class Blocked(FakeCollector):
        def search(self, query, location, limit):
            from jobmatch.collectors import BlockedError

            raise BlockedError("HTTP 999")

    collectors = {
        "104": FakeCollector("104", make_jobs("104", 30)),
        "cake": FakeCollector("cake", make_jobs("cake", 30)),
        "linkedin": Blocked("linkedin", []),
    }
    messages = []
    result = run_pipeline(
        resume,
        Settings(location="Taipei", top=10, candidates=20),
        FakeLLM(profile_dict),
        FakeEmbedder,
        collectors=collectors,
        progress=messages.append,
    )
    assert len(result.results) == 10
    assert "linkedin" in result.errors
    assert all(m.llm_score == 90 for m in result.results)
    assert any("continuing" in msg for msg in messages)
    md = to_markdown(result.results, result.profile)
    assert md.count("](https://example.com/") == 10

    from jobmatch.output import save

    paths = save(result.results, result.profile, tmp_path / "out")
    assert [p.name for p in paths] == ["results.md", "results.json", "results.html"]
    html = paths[2].read_text(encoding="utf-8")
    assert html.count('href="https://example.com/') == 10
    assert "fake reason" in html


def test_html_escapes_untrusted_text(profile_dict):
    from jobmatch.html_report import to_html
    from jobmatch.models import MatchResult

    job = Job(platform="104", job_id="1", title="<script>alert(1)</script>", company="A&B", url="https://x/1?a=1&b=2")
    html = to_html([MatchResult(job=job, llm_score=80, reason="ok")], CandidateProfile.model_validate(profile_dict))
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html and "A&amp;B" in html
