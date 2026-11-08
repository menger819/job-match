from jobmatch.collectors import cake, jobs_104, linkedin


def test_104_parse_search(fixtures_dir):
    jobs = jobs_104.parse_search((fixtures_dir / "104_search.json").read_text(encoding="utf-8"))
    assert len(jobs) == 2  # broken row skipped
    j = jobs[0]
    assert j.job_id == "8abcd"
    assert j.url == "https://www.104.com.tw/job/8abcd"
    assert j.company == "範例科技股份有限公司"
    assert j.title == "Python 後端工程師"
    assert j.salary == "60000~90000"
    assert jobs[1].salary == ""


def test_104_parse_detail(fixtures_dir):
    text = jobs_104.parse_detail((fixtures_dir / "104_detail.json").read_text(encoding="utf-8"))
    assert "RESTful API" in text
    assert "Skills: Python, Django" in text
    assert "AWS" in text


def test_104_area_code():
    assert jobs_104.area_code("Taipei") == "6001001000"
    assert jobs_104.area_code("台中市") == "6001008000"
    assert jobs_104.area_code("Remote") is None
    assert jobs_104.area_code(None) is None


def test_cake_parse_next_data(fixtures_dir):
    jobs = cake.parse_search((fixtures_dir / "cake_search_next.html").read_text(encoding="utf-8"))
    by_title = {j.title: j for j in jobs}
    assert set(by_title) == {"Senior Backend Engineer", "Machine Learning Engineer"}
    j = by_title["Senior Backend Engineer"]
    assert j.company == "Example Co."
    assert j.url == "https://www.cake.me/companies/example-co/jobs/senior-backend"
    assert j.location == "台北市"


def test_cake_parse_links_fallback(fixtures_dir):
    jobs = cake.parse_search((fixtures_dir / "cake_search_links.html").read_text(encoding="utf-8"))
    assert len(jobs) == 2  # duplicate link collapsed
    by_title = {j.title: j for j in jobs}
    assert by_title["Frontend Engineer"].company == "Example Co."
    assert by_title["Data Analyst"].url == "https://www.cake.me/companies/data-inc/jobs/data-analyst"


def test_cake_falls_back_to_listing_on_403(fixtures_dir):
    import httpx

    listing = (fixtures_dir / "cake_search_next.html").read_text(encoding="utf-8")

    class FakeClient:
        def __init__(self):
            self.urls = []

        def get_text(self, url, params=None, headers=None):
            self.urls.append((url, dict(params or {})))
            if url != "https://www.cake.me/jobs":
                req = httpx.Request("GET", url)
                raise httpx.HTTPStatusError("403", request=req, response=httpx.Response(403, request=req))
            return listing if params.get("page") == 1 else "<html></html>"

    client = FakeClient()
    jobs = cake.CakeCollector(client).search("Python", "Taipei", 10)
    assert len(jobs) == 2
    assert client.urls[0][0] == "https://www.cake.me/jobs/Python"
    assert "location_list[0]" not in client.urls[1][1]  # listing fallback is unfiltered


def test_linkedin_parse_search(fixtures_dir):
    jobs = linkedin.parse_search((fixtures_dir / "linkedin_search.html").read_text(encoding="utf-8"))
    assert [j.job_id for j in jobs] == ["4012345678", "4087654321"]
    j = jobs[0]
    assert j.title == "Backend Engineer"
    assert j.company == "Acme Corp"
    assert j.url == "https://www.linkedin.com/jobs/view/4012345678"
    assert j.location == "Taipei City, Taiwan"
    assert j.posted_at == "2026-11-01"


def test_linkedin_parse_detail():
    html = '<div class="show-more-less-html__markup"><p>Build APIs</p><ul><li>Python</li></ul></div>'
    assert linkedin.parse_detail(html) == "Build APIs\nPython"
