"""Step 1: read a resume file (.pdf / .docx / .txt / .md) into plain text."""

from __future__ import annotations

from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".markdown"}


class ResumeError(Exception):
    """Raised when a resume can't be read. The message is shown to the user."""


def load_resume(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise ResumeError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".doc":
        raise ResumeError(
            "Old Word format (.doc) is not supported. Open it in Word and use "
            "'Save As' -> .docx (or export as PDF), then try again."
        )
    if suffix not in SUPPORTED:
        raise ResumeError(f"Unsupported file type '{suffix}'. Use one of: .pdf, .docx, .txt, .md")

    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    text = _clean(text)
    if len(text) < 50:
        raise ResumeError(
            "Cannot read resume: almost no text was found. If this is a scanned PDF "
            "(an image), save your resume as .docx or .txt instead."
        )
    return text


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx  # python-docx

    document = docx.Document(str(path))
    lines = [p.text for p in document.paragraphs]
    # Many resumes put content in tables (two-column layouts).
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(lines)


def _clean(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    out, blank = [], 0
    for line in lines:
        if not line:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip()
