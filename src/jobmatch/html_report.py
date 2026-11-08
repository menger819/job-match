"""Render the top-N results as a single self-contained HTML page (no external files)."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import List

from jobmatch.models import CandidateProfile, MatchResult

PLATFORM_LABEL = {"104": "104", "cake": "Cake", "linkedin": "LinkedIn"}

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#111827;--muted:#6b7280;--border:#e5e7eb;
--accent:#6d4ae0;--good:#1f9d55;--ok:#d9891a;--low:#9ca3af;--chip:#eef0f3;
--p104:#ff7800;--pcake:#00b2a9;--plinkedin:#0a66c2}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b21;--text:#e5e7eb;
--muted:#9ca3af;--border:#2a2f38;--chip:#232831}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 -apple-system,"Segoe UI","PingFang TC","Noto Sans TC",Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 16px 48px}
h1{font-size:26px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 20px}
.profile{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px 18px;margin-bottom:22px}
.profile h2{font-size:15px;margin:0 0 6px}
.profile p{margin:0 0 10px;color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:13px}
ol{list-style:none;margin:0;padding:0;display:grid;gap:12px}
li{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px 18px;
display:grid;grid-template-columns:44px 1fr auto;gap:4px 14px;align-items:start}
.rank{grid-row:span 3;font-size:22px;font-weight:700;color:var(--muted);text-align:center;padding-top:2px}
.title{font-size:17px;font-weight:650;color:var(--text);text-decoration:none}
.title:hover{color:var(--accent);text-decoration:underline}
.company{color:var(--muted);grid-column:2}
.reason{grid-column:2 / span 2;margin:4px 0 0}
.meta{grid-column:2 / span 2;display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;font-size:13px}
.score{grid-row:span 2;text-align:center;min-width:64px}
.score b{display:block;font-size:22px;line-height:1.1}
.score small{color:var(--muted);font-size:11px}
.s-good{color:var(--good)}.s-ok{color:var(--ok)}.s-low{color:var(--low)}
.bar{height:5px;border-radius:3px;background:var(--chip);margin-top:6px;overflow:hidden}
.bar i{display:block;height:100%;background:currentColor}
.plat{color:#fff;border-radius:6px;padding:1px 8px;font-weight:600}
.p-104{background:var(--p104)}.p-cake{background:var(--pcake)}.p-linkedin{background:var(--plinkedin)}
footer{color:var(--muted);font-size:12px;margin-top:24px;text-align:center}
@media (max-width:520px){li{grid-template-columns:32px 1fr}.score{grid-column:2;grid-row:auto;text-align:left;display:flex;gap:8px;align-items:baseline}.bar{display:none}.reason,.meta{grid-column:1 / span 2}}
"""


def _score_class(score: float) -> str:
    return "s-good" if score >= 75 else "s-ok" if score >= 55 else "s-low"


def to_html(results: List[MatchResult], profile: CandidateProfile) -> str:
    e = escape
    items = []
    for i, m in enumerate(results, 1):
        j = m.job
        score = m.final_score
        meta = [f'<span class="plat p-{e(j.platform)}">{e(PLATFORM_LABEL.get(j.platform, j.platform))}</span>']
        if j.location:
            meta.append(f'<span class="chip">📍 {e(j.location[:60])}</span>')
        if j.salary:
            meta.append(f'<span class="chip">💰 {e(j.salary)}</span>')
        items.append(
            f"""<li>
  <div class="rank">{i}</div>
  <a class="title" href="{e(j.url)}" target="_blank" rel="noopener noreferrer">{e(j.title)}</a>
  <div class="score {_score_class(score)}"><b>{score:.0f}</b><small>match</small>
    <div class="bar"><i style="width:{max(0, min(100, score)):.0f}%"></i></div></div>
  <div class="company">{e(j.company or "(unknown company)")}</div>
  {f'<p class="reason">{e(m.reason)}</p>' if m.reason else ""}
  <div class="meta">{"".join(meta)}</div>
</li>"""
        )

    skills = "".join(f'<span class="chip">{e(s)}</span>' for s in profile.skills[:15])
    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Top {len(results)} Job Matches</title>
<style>{CSS}</style>
</head>
<body>
<main>
  <h1>🎯 Top {len(results)} Job Matches</h1>
  <p class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} · 104 / Cake / LinkedIn</p>
  <section class="profile">
    <h2>{e(profile.headline)}</h2>
    <p>{e(profile.summary)}</p>
    <div class="chips">{skills}</div>
  </section>
  <ol>
{"".join(items)}
  </ol>
  <footer>Scores are an AI judgment of fit (0–100). Always read the full job posting.</footer>
</main>
</body>
</html>
"""
