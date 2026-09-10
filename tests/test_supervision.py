"""Review providers never bypass opt-in, spend allowance, or structured validation."""

import json
import threading
import pytest
from locaii.storage import Store, digest
from locaii.supervision import Supervisor, validate_review, redact

REVIEW = {
    "verdict": "continue",
    "summary": "No observed output yet.",
    "plan": ["Create output", "Verify output"],
    "next_steps": ["Read the output before claiming completion."],
    "evidence_needed": ["Actual file receipt"],
}


def fixture(tmp_path):
    store = Store(tmp_path / "state")
    task = store.create_session("Test", tmp_path / "workspace")
    store.append(task["id"], "user", "Create a synthetic note and verify it.")
    return store, task["id"], Supervisor(store)


def test_remote_requires_opt_in(tmp_path):
    store, identifier, supervisor = fixture(tmp_path)
    with pytest.raises(PermissionError):
        supervisor.review(
            identifier, {"supervisor_provider": "openai"}, None, threading.Event()
        )


def test_budget_checked_before_provider_call(tmp_path):
    store, identifier, supervisor = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="allowance"):
        supervisor.review(
            identifier,
            {
                "supervisor_provider": "openai",
                "remote_enabled": True,
                "supervisor_token_budget": 1,
            },
            None,
            threading.Event(),
        )


def test_manual_packet_and_review_are_validated(tmp_path):
    store, identifier, supervisor = fixture(tmp_path)
    packet = json.loads(supervisor.export(identifier).read_text())
    assert packet["history_digest"] == digest(store.messages(identifier))
    file = tmp_path / "review.json"
    file.write_text(
        json.dumps({"history_digest": packet["history_digest"], "review": REVIEW})
    )
    assert supervisor.import_review(identifier, file) == REVIEW
    store.append(identifier, "user", "Changed request")
    with pytest.raises(ValueError, match="older task"):
        supervisor.import_review(identifier, file)


def test_malformed_review_cannot_claim_completion():
    with pytest.raises(Exception):
        validate_review({"verdict": "complete"})
    with pytest.raises(Exception):
        validate_review({**REVIEW, "execute": "rm -rf anywhere"})


def test_credential_redaction():
    assert "sk-" not in redact("token sk-abcdefghijk123456789")
    assert "secret123" not in redact("password=secret123")


def test_remote_allowance_includes_old_reviews_beyond_activity_tail(tmp_path):
    store, identifier, supervisor = fixture(tmp_path)
    store.event(
        identifier, "supervisor_usage", {"provider": "openai", "charged_tokens": 19999}
    )
    store.event(
        identifier, "supervisor_usage", {"provider": "local", "charged_tokens": 900000}
    )
    for _ in range(1100):
        store.event(identifier, "tool", {"name": "read_text_file", "status": "done"})
    assert store.supervisor_tokens(identifier, remote_only=True) == 19999
    assert store.supervisor_tokens(identifier) == 919999
    with pytest.raises(RuntimeError, match="allowance"):
        supervisor.review(
            identifier,
            {
                "supervisor_provider": "openai",
                "remote_enabled": True,
                "supervisor_token_budget": 20000,
            },
            None,
            threading.Event(),
        )


def test_packet_redaction_preserves_json_and_excludes_runtime_advice(tmp_path):
    store, identifier, supervisor = fixture(tmp_path)
    store.append(identifier, "user", "Inspect the configuration password=secret123")
    store.append(identifier, "user", "RUNTIME_ADVICE_SENTINEL", _runtime=True)
    packet = supervisor.packet(identifier)
    serialized = json.dumps(packet)
    assert "secret123" not in serialized
    assert "RUNTIME_ADVICE_SENTINEL" not in json.dumps(packet["user_requests"])
    assert "[REDACTED]" in serialized
    assert json.loads(supervisor.export(identifier).read_text()) == packet


def test_local_review_records_usage(tmp_path):
    store, identifier, supervisor = fixture(tmp_path)

    class Model:
        def chat(self, *args, **kwargs):
            return {
                "message": {"content": json.dumps(REVIEW)},
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 12,
                    "total_tokens": 20,
                },
            }

    assert (
        supervisor.review(
            identifier, {"supervisor_provider": "local"}, Model(), threading.Event()
        )
        == REVIEW
    )
    assert any(
        e["type"] == "supervisor_usage" and e["charged_tokens"] == 20
        for e in store.events(identifier)
    )


def test_claude_reviewer_disables_builtin_and_external_tools(tmp_path, monkeypatch):
    import locaii.supervision as module

    _, _, supervisor = fixture(tmp_path)
    captured = {}
    monkeypatch.setattr(module.shutil, "which", lambda _: "/synthetic/claude")

    class Process:
        returncode = 0

        def __init__(self, command, **kwargs):
            captured["command"] = command
            kwargs["stdout"].write(
                json.dumps(
                    {
                        "structured_output": REVIEW,
                        "usage": {"input_tokens": 20, "output_tokens": 30},
                    }
                )
            )
            kwargs["stdout"].flush()

        def poll(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    review, usage = supervisor._cli(
        "claude-cli", "", "Synthetic review packet.", threading.Event()
    )
    command = captured["command"]
    assert command[command.index("--tools") + 1] == ""
    assert "--safe-mode" in command
    assert "--strict-mcp-config" in command
    assert json.loads(command[command.index("--mcp-config") + 1]) == {"mcpServers": {}}
    assert review == REVIEW and usage["input_tokens"] == 20
