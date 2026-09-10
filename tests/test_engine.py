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
        [answer("", [repeated], "tool_calls"), answer("The target changed.")]
    )
    with pytest.raises(RuntimeError, match="unresolved tool failures"):
        run(Engine(store, model), identifier, "Continue")
    assert (workspace / "result.md").read_text() == "external revision"
    assert "changed or disappeared" in store.messages(identifier)[-2]["content"]


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
        options={"supervisor_provider": "local", "supervisor_every": 4},
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
        options={"supervisor_provider": "local"},
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
        ]
    )
    with pytest.raises(RuntimeError, match="unresolved tool failures"):
        run(
            Engine(store, models),
            identifier,
            "Continue",
            options={"allow_terminal": True},
        )
    assert not list((store.task_dir(identifier) / "jobs").glob("*.json"))
    results = [m for m in store.messages(identifier) if m["role"] == "tool"]
    assert "unknown" in results[-1]["content"]
