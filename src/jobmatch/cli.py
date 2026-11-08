"""Command line entry point: `jobmatch <resume> [options]`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from jobmatch.collectors import REGISTRY


def _parse_platforms(value: str) -> List[str]:
    names = [p.strip().lower() for p in value.split(",") if p.strip()]
    bad = [p for p in names if p not in REGISTRY]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown platform(s): {', '.join(bad)}. Choose from 104, cake, linkedin")
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jobmatch",
        description="Find the top matching jobs on 104, Cake and LinkedIn for your resume.",
    )
    p.add_argument("resume", type=Path, help="Path to your resume (.pdf, .docx, .txt, .md)")
    p.add_argument("--location", help="Job location, e.g. Taipei, Taichung, Remote (default: JOBMATCH_LOCATION)")
    p.add_argument("--top", type=int, default=10, help="How many jobs to show (default: 10)")
    p.add_argument("--platforms", type=_parse_platforms, help="Comma-separated: 104,cake,linkedin (default: all)")
    p.add_argument("--max-jobs-per-platform", type=int, default=60, help="Jobs to collect per platform (default: 60)")
    p.add_argument("--candidates", type=int, default=40, help="Jobs sent to the AI reranker (default: 40)")
    p.add_argument("--no-cache", action="store_true", help="Ignore saved search results and search fresh")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Where to save results (default: output/)")
    p.add_argument("--verbose", action="store_true", help="Show detailed logs")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        import warnings

        warnings.filterwarnings("ignore")  # e.g. urllib3 LibreSSL notice on macOS system Python
        for noisy in ("httpx", "httpcore", "sentence_transformers", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    from jobmatch.llm import ClaudeCodeLLM, LLMError, make_llm
    from jobmatch.output import save, to_terminal
    from jobmatch.pipeline import Settings, run_pipeline
    from jobmatch.resume_loader import ResumeError
    from jobmatch.retrieval import SentenceTransformerEmbedder

    platforms = args.platforms or ["104", "cake", "linkedin"]
    if os.getenv("LINKEDIN_ENABLED", "true").lower() in ("false", "0", "no") and "linkedin" in platforms:
        platforms = [p for p in platforms if p != "linkedin"]
        if args.platforms and "linkedin" in args.platforms:
            print("Note: LinkedIn is disabled by LINKEDIN_ENABLED=false in .env")

    settings = Settings(
        location=args.location or os.getenv("JOBMATCH_LOCATION") or None,
        top=args.top,
        platforms=platforms,
        max_jobs_per_platform=args.max_jobs_per_platform,
        candidates=max(args.candidates, args.top),
        use_cache=not args.no_cache,
    )

    try:
        llm = make_llm()
        engine = "Claude Code login" if isinstance(llm, ClaudeCodeLLM) else "Anthropic API key"
        print(f"AI engine: {engine}")
        result = run_pipeline(args.resume, settings, llm, SentenceTransformerEmbedder)
    except (ResumeError, LLMError) as e:
        print(f"\n❌ {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130

    if not result.results:
        print("\n0 jobs found. Try a different --location, fewer words in your resume headline, or --no-cache.")
        return 2

    print(f"\n🎯 Top {len(result.results)} jobs:\n")
    print(to_terminal(result.results))
    paths = save(result.results, result.profile, args.output_dir)
    print(f"\nSaved: {', '.join(str(p) for p in paths)}")
    u = llm.usage
    print(f"Tokens used: {u['input']:,} in / {u['output']:,} out (cache read {u['cache_read']:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
