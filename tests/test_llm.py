import json
import subprocess

import pytest

from jobmatch import llm as llm_mod


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    path = tmp_path / "claude"
    path.write_text("")
    monkeypatch.setenv("JOBMATCH_CLAUDE_PATH", str(path))
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("JOBMATCH_LLM", raising=False)
    return str(path)


def test_placeholder_key_selects_claude_code(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", llm_mod.PLACEHOLDER_KEY)
    assert not llm_mod.has_api_key()
    engine = llm_mod.make_llm()
    assert isinstance(engine, llm_mod.ClaudeCodeLLM)
    assert engine.cli == fake_cli


def test_real_key_selects_api(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    assert llm_mod.has_api_key()
    assert isinstance(llm_mod.make_llm(), llm_mod.ClaudeJSONLLM)


def test_claude_code_parses_structured_output(fake_cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", llm_mod.PLACEHOLDER_KEY)
    seen = {}

    def fake_run(cmd, input, env, **kwargs):
        seen.update(cmd=cmd, input=input, env=env)
        out = {"type": "result", "subtype": "success", "is_error": False,
               "structured_output": {"name": "Jane"}, "usage": {"input_tokens": 5, "output_tokens": 7}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    engine = llm_mod.ClaudeCodeLLM()
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = engine.complete_json("sys", [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}], schema)
    assert result == {"name": "Jane"}
    assert seen["input"] == "hello\n\nworld"
    assert "--json-schema" in seen["cmd"] and "-p" in seen["cmd"]
    assert "ANTHROPIC_API_KEY" not in seen["env"]  # uses the Claude Code login
    assert engine.usage["output"] == 7


def test_claude_code_error_is_reported(fake_cli, monkeypatch):
    def fake_run(cmd, **kwargs):
        out = {"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "usage limit reached"}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(llm_mod.LLMError, match="usage limit"):
        llm_mod.ClaudeCodeLLM().complete_json("s", [{"type": "text", "text": "x"}], {"type": "object"})
