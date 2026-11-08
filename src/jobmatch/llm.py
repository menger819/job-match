"""Thin wrapper around the Claude API that returns schema-validated JSON.

Used by both LLM steps: resume profiling (technique 1) and reranking (technique 3).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Protocol

DEFAULT_MODEL = "claude-opus-5-5"


class LLMError(Exception):
    """Raised when the model can't return a usable answer. Message is user-facing."""


class JSONLLM(Protocol):
    """Anything that can answer a prompt with JSON matching a schema (lets tests use a fake)."""

    def complete_json(
        self,
        system: str,
        content: List[Dict[str, Any]],
        schema: Dict[str, Any],
        max_tokens: int = 16000,
    ) -> Dict[str, Any]: ...


class ClaudeJSONLLM:
    def __init__(self, model: Optional[str] = None, effort: str = "medium"):
        import anthropic

        if not has_api_key():
            raise LLMError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and paste your key "
                "(see README, Operation Manual Step 7)."
            )
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("JOBMATCH_MODEL") or DEFAULT_MODEL
        self.effort = effort
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def complete_json(self, system, content, schema, max_tokens=16000):
        a = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
                # If a safety classifier declines, the API retries on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.AuthenticationError as e:
            raise LLMError("Your ANTHROPIC_API_KEY was rejected. Check the key in .env.") from e
        except a.NotFoundError as e:
            raise LLMError(f"Model '{self.model}' not found. Check JOBMATCH_MODEL in .env.") from e
        except a.RateLimitError as e:
            raise LLMError("Anthropic API rate limit reached. Wait a minute and try again.") from e
        except a.APIStatusError as e:
            raise LLMError(f"Anthropic API error ({e.status_code}): {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API. Check your internet connection.") from e

        self._track(response.usage)

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The model's answer was cut off (max_tokens). Try fewer jobs per batch.")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("The model returned no text.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("The model returned invalid JSON.") from e

    def _track(self, usage) -> None:
        if usage is None:
            return
        self.usage["input"] += getattr(usage, "input_tokens", 0) or 0
        self.usage["output"] += getattr(usage, "output_tokens", 0) or 0
        self.usage["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.usage["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0


PLACEHOLDER_KEY = "sk-ant-put-your-key-here"


def has_api_key() -> bool:
    key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    return bool(key) and key != PLACEHOLDER_KEY or bool(os.getenv("ANTHROPIC_AUTH_TOKEN"))


def find_claude_cli() -> Optional[str]:
    """Locate the Claude Code program: PATH, ~/.claude/local, or the VS Code / Cursor extension."""
    import glob
    import shutil

    explicit = os.getenv("JOBMATCH_CLAUDE_PATH")
    if explicit and os.path.exists(explicit):
        return explicit
    found = shutil.which("claude")
    if found:
        return found
    home = os.path.expanduser("~")
    candidates = [os.path.join(home, ".claude", "local", "claude")]
    for ext_dir in (".vscode/extensions", ".cursor/extensions", ".vscode-insiders/extensions"):
        pattern = os.path.join(home, ext_dir, "anthropic.claude-code-*", "resources", "native-binary", "claude")
        candidates += sorted(glob.glob(pattern), key=_version_key, reverse=True)
    return next((c for c in candidates if os.path.exists(c)), None)


def _version_key(path: str):
    import re

    m = re.search(r"claude-code-(\d+)\.(\d+)\.(\d+)", path)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


class ClaudeCodeLLM:
    """Runs Claude through the Claude Code program (`claude -p`), using your Claude Code login.

    No API key needed; usage counts toward your Claude plan's limits.
    """

    def __init__(self, cli_path: Optional[str] = None, model: Optional[str] = None, effort: str = "medium"):
        self.cli = cli_path or find_claude_cli()
        if not self.cli:
            raise LLMError(
                "No ANTHROPIC_API_KEY and no Claude Code program found. Either add an API key to .env, "
                "or install Claude Code (https://claude.com/claude-code) and log in once."
            )
        self.model = model or os.getenv("JOBMATCH_MODEL") or None
        self.effort = effort
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def complete_json(self, system, content, schema, max_tokens=16000):
        import subprocess
        import tempfile

        prompt = "\n\n".join(block["text"] for block in content if block.get("type") == "text")
        cmd = [
            self.cli, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(schema),
            "--system-prompt", system,
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--effort", self.effort,
        ]
        if self.model:
            cmd += ["--model", self.model]

        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)  # make Claude Code use its own login, not a placeholder key
        try:
            # Run outside the project so no project settings or CLAUDE.md are picked up.
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=600,
                env=env, cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired as e:
            raise LLMError("Claude Code took too long to answer (10 min). Try again.") from e

        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            detail = (proc.stderr or proc.stdout).strip()[:300]
            if "login" in detail.lower() or "auth" in detail.lower():
                raise LLMError("Claude Code is not logged in. Open Claude Code once and log in, then retry.") from e
            raise LLMError(f"Claude Code returned an unexpected answer: {detail}") from e

        usage = data.get("usage") or {}
        self.usage["input"] += usage.get("input_tokens", 0) or 0
        self.usage["output"] += usage.get("output_tokens", 0) or 0
        self.usage["cache_read"] += usage.get("cache_read_input_tokens", 0) or 0
        self.usage["cache_write"] += usage.get("cache_creation_input_tokens", 0) or 0

        if data.get("is_error") or data.get("subtype") != "success":
            msg = str(data.get("result") or data.get("subtype") or "unknown error")[:300]
            if "limit" in msg.lower():
                raise LLMError(f"Claude plan usage limit reached: {msg}")
            raise LLMError(f"Claude Code error: {msg}")
        output = data.get("structured_output")
        if not isinstance(output, dict):
            raise LLMError("Claude Code returned no structured answer.")
        return output


def make_llm() -> "JSONLLM":
    """Pick the AI engine: JOBMATCH_LLM=api | claude-code | auto (default)."""
    choice = (os.getenv("JOBMATCH_LLM") or "auto").strip().lower()
    if choice == "api" or (choice == "auto" and has_api_key()):
        return ClaudeJSONLLM()
    if choice in ("claude-code", "auto"):
        return ClaudeCodeLLM()
    raise LLMError(f"Unknown JOBMATCH_LLM '{choice}'. Use auto, api or claude-code.")


def strict_object(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Build a JSON-schema object where every property is required (structured-output friendly)."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }
