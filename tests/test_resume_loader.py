import pytest

from jobmatch.resume_loader import ResumeError, load_resume

RESUME = "Jane Chen\nSenior Backend Engineer\n\n\n\nSkills: Python, Django, AWS, PostgreSQL, Docker, Kubernetes"


def test_txt_and_md(tmp_path):
    for ext in (".txt", ".md"):
        p = tmp_path / f"r{ext}"
        p.write_text(RESUME, encoding="utf-8")
        text = load_resume(p)
        assert "Senior Backend Engineer" in text
        assert "\n\n\n" not in text  # blank lines collapsed


def test_docx(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    for line in RESUME.split("\n"):
        d.add_paragraph(line)
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text = "Company"
    t.cell(0, 1).text = "Acme 2020-2026"
    p = tmp_path / "r.docx"
    d.save(str(p))
    text = load_resume(p)
    assert "Kubernetes" in text
    assert "Acme 2020-2026" in text


def test_pdf(tmp_path):
    pytest.importorskip("pypdf")
    p = tmp_path / "r.pdf"
    p.write_bytes(_minimal_pdf("Senior Backend Engineer with Python Django AWS PostgreSQL experience"))
    assert "Backend Engineer" in load_resume(p)


def test_errors(tmp_path):
    with pytest.raises(ResumeError, match="not found"):
        load_resume(tmp_path / "missing.pdf")
    doc = tmp_path / "old.doc"
    doc.write_bytes(b"x")
    with pytest.raises(ResumeError, match=".docx"):
        load_resume(doc)
    short = tmp_path / "short.txt"
    short.write_text("hi")
    with pytest.raises(ResumeError, match="Cannot read resume"):
        load_resume(short)


def _minimal_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out
