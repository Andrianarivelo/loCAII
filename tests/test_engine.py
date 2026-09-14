"""Durable agent-loop regressions with deterministic models and synthetic workspaces."""

from __future__ import annotations
import copy
import json
import threading
from pathlib import Path
import pytest
from locaii.engine import Engine, clean_messages
from locaii.models import ModelCancelled
from locaii.storage import Store, canonical, digest


def answer(text="Done.", calls=None, finish="stop"):
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return {
        "message": message,
        "finish_reason": finish,
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    }


def call(name, args, identifier):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class FakeModels:
    """Return scripted inference results while recording every actual prompt."""

    def __init__(self, responses=(), context=32768):
        self.responses = list(responses)
        self.prompts = []
        self.context = context
        self.started = []
        self.memory_response = answer(
            "# Task memory\n\nOriginal objective retained. Verified synthetic tool receipts; no pending changes."
        )

    def list_models(self):
        return [{"id": "fixture", "name": "Fixture", "context": self.context}]

    def start(self, identifier, **kwargs):
        self.started.append(identifier)

    def count_tokens(self, messages, tools=None):
        return len(canonical(messages)) // 4 + 100

    def chat(self, messages, tools=None, **kwargs):
        self.prompts.append(copy.deepcopy(messages))
        if messages and messages[0]["content"].startswith(
            "Write concise Markdown task memory"
        ):
            return copy.deepcopy(self.memory_response)
        if not self.responses:
            raise AssertionError("Unexpected extra inference call")
        result = self.responses.pop(0)
        if callable(result):
            result = result(messages, kwargs)
        if isinstance(result, BaseException):
            raise result
        return copy.deepcopy(result)


class FakeSupervisor:
    def __init__(self, reviews):
        self.reviews = list(reviews)
        self.calls = 0

    def review(self, *args):
        self.calls += 1
        return self.reviews.pop(0)


@pytest.fixture
def task(tmp_path):
    store = Store(tmp_path / "state")
    session = store.create_session("Synthetic task", tmp_path / "workspace")
    return store, session["id"], Path(session["workspace"])


def run(engine, identifier, text, **extra):
    options = {
        "model_id": "fixture",
        "max_iterations": 10,
        "workspace_writes": True,
        "allow_terminal": False,
        "read_all_files": False,
    }
    options.update(extra.pop("options", {}))
    return engine.run(identifier, text, options=options, **extra)


def test_native_multitool_writes_have_real_receipts(task):
    store, identifier, workspace = task
    models = FakeModels(
        [
            answer(
                "",
                [
                    call("write_file", {"path": "a.txt", "content": "alpha"}, "a"),
                    call("write_file", {"path": "b.txt", "content": "beta"}, "b"),
                ],
                "tool_calls",
            ),
            answer(
                "", [call("read_text_file", {"path": "a.txt"}, "read")], "tool_calls"
            ),
            answer("Created both files and checked alpha."),
        ]
    )
    result = run(
        Engine(store, models), identifier, "Create a.txt and b.txt, then verify a.txt."
    )
    assert result["status"] == "completed"
    assert (workspace / "a.txt").read_text() == "alpha"
    assert (workspace / "b.txt").read_text() == "beta"
    receipts = [
        json.loads(m["content"])
        for m in store.messages(identifier)
        if m["role"] == "tool"
    ]
    assert len(receipts) == 3
    assert receipts[2]["content"] == "alpha"
    assert len(models.prompts[1][-3:]) == 3


def test_resume_retains_original_objective_and_completed_effect(task):
    store, identifier, workspace = task
    request = call("write_file", {"path": "result.md", "content": "verified"}, "first")
    original = "Create result.md containing verified, then inspect it."
    first = FakeModels([answer("", [request], "tool_calls")])
    with pytest.raises(RuntimeError, match="Iteration limit"):
        run(Engine(store, first), identifier, original, options={"max_iterations": 1})
    stamp = (workspace / "result.md").stat().st_mtime_ns
    second = FakeModels(
        [
            answer("", [dict(request, id="retry")], "tool_calls"),
            answer("Verified result.md."),
        ]
    )
    second.memory_response = answer("", finish="length")
    result = run(Engine(store, second), identifier, "Continue")
    assert result["status"] == "completed"
    assert (workspace / "result.md").stat().st_mtime_ns == stamp
    receipts = [
        json.loads(m["content"])
        for m in store.messages(identifier)
        if m["role"] == "tool"
    ]
    assert receipts[-1]["reused_receipt"] is True
    assert any(m.get("content") == original for m in second.prompts[0])
    assert original in store.memory(
        identifier
    ), "Continue must not replace the task objective in memory"


def test_empty_responses_are_bounded_and_never_complete(task):
    store, identifier, _ = task
    models = FakeModels([answer(""), answer(""), answer("")])
    with pytest.raises(RuntimeError, match="two recovery attempts"):
        run(Engine(store, models), identifier, "Give a visible answer.")
    assert len(models.prompts) == 3
    assert store.get_session(identifier)["status"] == "error"
    assert not any(event["type"] == "done" for event in store.events(identifier))


def test_resume_write_receipt_does_not_hide_external_file_change(task):
    store, identifier, workspace = task
    request = call(
        "write_file", {"path": "result.md", "content": "original result"}, "first"
    )
    with pytest.raises(RuntimeError, match="Iteration limit"):
        run(
            Engine(store, FakeModels([answer("", [request], "tool_calls")])),
            identifier,
            "Create a separate result.",
            options={"max_iterations": 1},
        )
    (workspace / "result.md").write_text("external revision")
    repeated = dict(request, id="repeat")
    model = FakeModels(
        [
            answer("", [repeated], "tool_calls"),
            answer("The target changed."),
            answer("The target changed; I could not verify the write."),
        ]
    )
    result = run(Engine(store, model), identifier, "Continue")
    # The partial answer is kept, but the unresolved failure is reported with it
    # instead of discarding the turn; the external revision is never overwritten.
    assert result["status"] == "completed"
    assert result["unresolved_tool_failures"][0]["tool"] == "write_file"
    assert any(
        "unresolved" in m.get("content", "")
        for m in model.prompts[1] + model.prompts[2]
    )
    assert (workspace / "result.md").read_text() == "external revision"
    assert any(
        "changed or disappeared" in str(m.get("content", ""))
        for m in store.messages(identifier)
    )


def test_incomplete_answer_continues_without_false_completion(task):
    store, identifier, _ = task
    models = FakeModels(
        [answer("Part one", finish="length"), answer("Part two; complete.")]
    )
    result = run(Engine(store, models), identifier, "Explain both parts.")
    assert result["iterations"] == 2
    assert result["text"] == "Part two; complete."
    assert any(
        "Continue the incomplete response" in m.get("content", "")
        for m in models.prompts[1]
    )


def test_archive_restart_reuses_checkpoint_and_keeps_user_constraints(task):
    store, identifier, _ = task
    original = "Inspect the synthetic log, preserve all originals, then summarize."
    store.append(identifier, "user", original)
    store.append(
        identifier,
        answer("", [call("read_text_file", {"path": "huge.log"}, "read")])["message"],
    )
    store.append(
        identifier,
        "tool",
        "START " + "large receipt " * 20000 + " END_MARKER",
        tool_call_id="read",
    )
    models = FakeModels(context=16000)
    engine = Engine(store, models)
    compacted, count = engine._compact(
        identifier,
        engine._context(identifier),
        [],
        lambda *args: None,
        threading.Event(),
        16000,
    )
    assert count < 16000 - 4096
    checkpoint = store.checkpoint(identifier)
    archive = json.loads(Path(checkpoint["archive"]).read_text())
    assert "END_MARKER" in archive["messages"][-1]["content"]
    assert original in canonical(compacted)
    store.append(identifier, "user", "Continue")
    restored = Engine(store, models)._context(identifier)
    assert original in canonical(restored)
    assert "large receipt " * 1000 not in canonical(restored)
    assert any(m.get("content") == "Continue" for m in restored)


def test_changed_middle_of_archived_prefix_invalidates_checkpoint(task):
    store, identifier, _ = task
    store.append(identifier, "user", "Inspect and summarize.")
    store.append(
        identifier,
        answer("", [call("read_text_file", {"path": "a"}, "read")])["message"],
    )
    original = "A" * 100000 + "ORIGINAL_MIDDLE" + "Z" * 100000
    store.append(identifier, "tool", original, tool_call_id="read")
    models = FakeModels(context=16000)
    engine = Engine(store, models)
    engine._compact(
        identifier,
        engine._context(identifier),
        [],
        lambda *args: None,
        threading.Event(),
        16000,
    )
    messages = store.messages(identifier)
    messages[-1]["content"] = original.replace("ORIGINAL_MIDDLE", "CHANGED_MIDDLE")
    with store.db:
        row = store.db.execute(
            "SELECT id FROM messages WHERE session_id=? ORDER BY id DESC LIMIT 1",
            (identifier,),
        ).fetchone()
        store.db.execute(
            "UPDATE messages SET payload=? WHERE id=?",
            (canonical(messages[-1]), row[0]),
        )
    restored = engine._context(identifier)
    assert "CHANGED_MIDDLE" in canonical(restored)
    assert len(canonical(restored)) > 200000


def test_cancellation_after_tool_request_preserves_unknown_outcome(task):
    store, identifier, workspace = task
    cancel = threading.Event()

    def cancel_before_tool(messages, kwargs):
        cancel.set()
        return answer(
            "",
            [
                call(
                    "write_file",
                    {"path": "should_not_exist.txt", "content": "x"},
                    "pending",
                )
            ],
            "tool_calls",
        )

    models = FakeModels([cancel_before_tool])
    result = run(
        Engine(store, models), identifier, "Create a synthetic file.", cancel=cancel
    )
    assert result["status"] == "interrupted"
    assert not (workspace / "should_not_exist.txt").exists()
    assert any(m.get("tool_calls") for m in store.messages(identifier))
    restored = clean_messages(store.messages(identifier))
    assert "unknown" in restored[-1]["content"].lower()


def test_backend_cancellation_is_interruption_not_error(task):
    store, identifier, _ = task
    cancel = threading.Event()

    def cancelled_backend(messages, kwargs):
        cancel.set()
        raise ModelCancelled("Model operation cancelled")

    models = FakeModels([cancelled_backend])
    result = run(
        Engine(store, models), identifier, "Remember this objective.", cancel=cancel
    )
    assert result["status"] == "interrupted"
    assert store.get_session(identifier)["status"] == "interrupted"
    assert store.messages(identifier)[0]["content"] == "Remember this objective."


def test_supervisor_continue_causes_local_iteration_before_done(task):
    store, identifier, workspace = task
    supervisor = FakeSupervisor(
        [
            {
                "verdict": "continue",
                "summary": "Initial plan: inspect evidence first",
                "next_steps": ["Read evidence.txt."],
            },
            {
                "verdict": "continue",
                "summary": "Missing verification",
                "next_steps": ["Read evidence.txt."],
            },
            {"verdict": "complete", "summary": "Evidence verified", "next_steps": []},
        ]
    )
    (workspace / "evidence.txt").write_text("VERIFIED")
    models = FakeModels(
        [
            answer("I am done."),
            answer(
                "",
                [call("read_text_file", {"path": "evidence.txt"}, "check")],
                "tool_calls",
            ),
            answer("Verified the requested evidence."),
        ]
    )
    result = run(
        Engine(store, models, supervisor),
        identifier,
        "Verify evidence.txt before finishing.",
        options={
            "supervisor_provider": "local",
            "supervisor_every": 4,
            "supervisor_scope": "all",
        },
    )
    assert result["status"] == "completed"
    assert result["iterations"] == 3
    assert supervisor.calls == 3
    events = store.events(identifier)
    assert [event["type"] for event in events].count("done") == 1
    assert [event["verdict"] for event in events if event["type"] == "review"] == [
        "continue",
        "continue",
        "complete",
    ]
    assert "Missing verification" in canonical(models.prompts[1])


def test_compaction_memory_write_failure_does_not_publish_cursor(task, monkeypatch):
    store, identifier, _ = task
    store.append(identifier, "user", "Inspect and summarize.")
    store.append(
        identifier,
        answer("", [call("read_text_file", {"path": "a"}, "read")])["message"],
    )
    store.append(identifier, "tool", "receipt " * 30000, tool_call_id="read")
    models = FakeModels(context=16000)
    engine = Engine(store, models)
    before = store.messages(identifier)
    monkeypatch.setattr(
        store, "save_memory", lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        engine._compact(
            identifier,
            engine._context(identifier),
            [],
            lambda *args: None,
            threading.Event(),
            16000,
        )
    assert (
        store.checkpoint(identifier) == {}
    ), "Cursor must not precede its matching durable memory"
    assert store.messages(identifier) == before


def test_initial_supervisor_plan_reaches_first_local_prediction(task):
    store, identifier, _ = task
    supervisor = FakeSupervisor(
        [
            {
                "verdict": "continue",
                "summary": "INITIAL_PLAN_SENTINEL",
                "next_steps": ["Answer concisely."],
            },
            {
                "verdict": "complete",
                "summary": "Answer is sufficient.",
                "next_steps": [],
            },
        ]
    )

    def inspect_initial_prompt(messages, kwargs):
        assert "INITIAL_PLAN_SENTINEL" in canonical(messages)
        assert "Original user objective." in canonical(messages)
        return answer("A concise verified answer.")

    models = FakeModels([inspect_initial_prompt])
    result = run(
        Engine(store, models, supervisor),
        identifier,
        "Original user objective.",
        options={"supervisor_provider": "local", "supervisor_scope": "all"},
    )
    assert result["status"] == "completed"
    assert result["iterations"] == 1
    assert supervisor.calls == 2


def test_resume_does_not_repeat_command_with_unobserved_outcome(task):
    store, identifier, workspace = task
    store.append(identifier, "user", "Run the synthetic command once.")
    store.append(
        identifier,
        answer("", [call("run_command", {"command": "echo never-repeat"}, "unknown")])[
            "message"
        ],
    )
    models = FakeModels(
        [
            answer(
                "",
                [call("run_command", {"command": "echo never-repeat"}, "repeat")],
                "tool_calls",
            ),
            answer("Cannot verify the previous command yet."),
            answer("The earlier command outcome is unknown and was not repeated."),
        ]
    )
    result = run(
        Engine(store, models),
        identifier,
        "Continue",
        options={"allow_terminal": True},
    )
    assert result["unresolved_tool_failures"][0]["tool"] == "run_command"
    assert not list((store.task_dir(identifier) / "jobs").glob("*.json"))
    results = [m for m in store.messages(identifier) if m["role"] == "tool"]
    assert "unknown" in results[-1]["content"]


def test_fallible_summary_cannot_replace_recorded_resume_evidence(task):
    """A summary with stale status or mistyped paths keeps exact current receipts."""
    store, identifier, workspace = task
    store.append(identifier, "user", "Write a synthetic result and verify it.")
    store.append(identifier, "user", "Continue. Read result.md without rewriting.")
    store.append(
        identifier,
        "tool",
        '{"path":"result.md","content":"BIRCH-629"}',
        tool_call_id="receipt",
    )
    store.append(identifier, "assistant", "RESUME_READY BIRCH-629")
    models = FakeModels()
    models.memory_response = answer(
        "# Memory\nPending: write /mistyped/workspace/result.md again."
    )
    engine = Engine(store, models)
    engine._memory(identifier, threading.Event(), lambda *_: None)
    memory = store.memory(identifier)
    recorded, interpretation = memory.split("## Model interpretation (fallible)")
    assert str(workspace) in recorded
    assert "Continue. Read result.md without rewriting." in recorded
    assert "RESUME_READY BIRCH-629" in recorded
    assert r'"path\":\"result.md\"' in recorded
    assert "/mistyped/" not in recorded
    assert "/mistyped/" in interpretation
    assert "RESUME_READY BIRCH-629" in canonical(models.prompts[-1])


def test_saved_compaction_evidence_survives_later_memory_replacement(task):
    """A later model or tool memory update cannot erase adopted archive evidence."""
    store, identifier, _ = task
    store.append(identifier, "user", "Summarize the synthetic log.")
    store.append(
        identifier,
        answer("", [call("read_text_file", {"path": "synthetic.log"}, "read")])[
            "message"
        ],
    )
    store.append(
        identifier,
        "tool",
        "OBSERVED_MARKER_629 " + "receipt " * 20000,
        tool_call_id="read",
    )
    models = FakeModels(context=16000)
    engine = Engine(store, models)
    engine._compact(
        identifier,
        engine._context(identifier),
        [],
        lambda *_: None,
        threading.Event(),
        16000,
    )
    store.save_memory(identifier, "A model summary incorrectly omitted all receipts.")
    restored = canonical(Engine(store, models)._context(identifier))
    assert "OBSERVED_MARKER_629" in restored
    assert "incorrectly omitted" not in restored


def test_plain_chat_ignores_review_budget_and_old_tool_history(task):
    """A new conversational turn uses no reviewer, even in an old work session."""
    store, identifier, _ = task
    store.append(identifier, "user", "Earlier file task")
    store.append(
        identifier,
        answer("", [call("list_directory", {"path": "."}, "old")])["message"],
    )
    store.append(
        identifier, "tool", "Old listing", name="list_directory", tool_call_id="old"
    )
    supervisor = FakeSupervisor([])
    models = FakeModels([answer("Hello. How can I help?")])
    result = run(
        Engine(store, models, supervisor),
        identifier,
        "hello",
        options={
            "supervisor_provider": "codex-cli",
            "remote_enabled": True,
            "supervisor_token_budget": 1,
        },
    )
    assert result["status"] == "completed"
    assert supervisor.calls == 0
    assert not any(
        e["type"] in {"review", "review_skipped", "error"}
        for e in store.events(identifier)
    )


@pytest.mark.parametrize("scope,every", [("all", 1), ("tasks", 1), ("tasks", 4)])
def test_allowance_exhaustion_keeps_local_work_and_skips_once(task, scope, every):
    """Preplan, milestone and final review caps never undo a completed local task."""
    from locaii.supervision import Supervisor

    store, identifier, workspace = task
    models = FakeModels(
        [
            answer(
                "",
                [
                    call(
                        "write_file",
                        {"path": "derived.txt", "content": "verified"},
                        "w",
                    )
                ],
                "tool_calls",
            ),
            answer(
                "", [call("read_text_file", {"path": "derived.txt"}, "r")], "tool_calls"
            ),
            answer("Read back verified from derived.txt."),
        ]
    )
    result = run(
        Engine(store, models, Supervisor(store)),
        identifier,
        "Write and verify a synthetic derived file.",
        options={
            "supervisor_provider": "codex-cli",
            "supervisor_scope": scope,
            "supervisor_every": every,
            "remote_enabled": True,
            "supervisor_token_budget": 1,
        },
    )
    assert result["status"] == "completed"
    assert result["review_status"] == "skipped_allowance"
    assert (workspace / "derived.txt").read_text() == "verified"
    assert store.get_session(identifier)["status"] == "completed"
    events = store.events(identifier)
    skipped = [e for e in events if e["type"] == "review_skipped"]
    assert len(skipped) == 1 and skipped[0]["reviewed"] is False
    assert skipped[0]["allowance_tokens"] == 1
    assert not any(
        e["type"] in {"error", "supervisor_usage", "supervisor_review"} for e in events
    )
    assert store.supervisor_tokens(identifier) == 0


def test_default_tool_scope_still_reviews_and_requests_correction(task):
    """Tools-only mode preserves evaluation and bounded follow-up iterations."""
    store, identifier, workspace = task
    (workspace / "evidence.txt").write_text("VERIFIED")
    supervisor = FakeSupervisor(
        [
            {
                "verdict": "continue",
                "summary": "Read the evidence again to confirm",
                "next_steps": ["Read evidence.txt."],
            },
            {
                "verdict": "complete",
                "summary": "Observed verification",
                "next_steps": [],
            },
        ]
    )
    models = FakeModels(
        [
            answer(
                "",
                [call("read_text_file", {"path": "evidence.txt"}, "one")],
                "tool_calls",
            ),
            answer("Read the evidence."),
            answer(
                "",
                [call("read_text_file", {"path": "evidence.txt"}, "two")],
                "tool_calls",
            ),
            answer("Verified twice."),
        ]
    )
    result = run(
        Engine(store, models, supervisor),
        identifier,
        "Verify evidence.txt.",
        options={"supervisor_provider": "local"},
    )
    assert result["status"] == "completed" and supervisor.calls == 2
    assert result["tool_rounds"] == 2


def test_optional_review_does_not_swallow_provider_errors(task):
    """Authentication/provider failures stay visible, unlike an explicit budget cap."""
    store, identifier, _ = task

    class BrokenSupervisor:
        def review(self, *args):
            raise RuntimeError("Provider unavailable")

    with pytest.raises(RuntimeError, match="Provider unavailable"):
        run(
            Engine(store, FakeModels([]), BrokenSupervisor()),
            identifier,
            "Hello",
            options={
                "supervisor_provider": "codex-cli",
                "supervisor_scope": "all",
            },
        )


def test_allowance_after_successful_plan_preserves_completed_answer(task):
    """Reproduce a funded preplan followed by an unaffordable final review."""
    from locaii.supervision import SupervisorBudgetExceeded

    store, identifier, _ = task

    class PlanThenCap:
        calls = 0

        def review(self, *args):
            self.calls += 1
            if self.calls == 1:
                return {
                    "verdict": "continue",
                    "summary": "Answer concisely",
                    "next_steps": [],
                }
            raise SupervisorBudgetExceeded(14000, 21000, 24000)

    supervisor = PlanThenCap()
    result = run(
        Engine(store, FakeModels([answer("A complete local answer.")]), supervisor),
        identifier,
        "A simple question",
        options={
            "supervisor_provider": "codex-cli",
            "supervisor_scope": "all",
            "remote_enabled": True,
            "supervisor_token_budget": 24000,
        },
    )
    assert (
        result["status"] == "completed" and result["text"] == "A complete local answer."
    )
    assert result["review_status"] == "skipped_allowance"
    assert supervisor.calls == 2
    assert not any(e["type"] == "error" for e in store.events(identifier))
