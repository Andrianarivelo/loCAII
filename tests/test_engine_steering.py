"""Live steering uses deterministic arrivals at actual inference/tool boundaries."""

import json
import threading

from test_engine import FakeModels, answer, call, run, task
from locaii.engine import Engine
from locaii.storage import Store, canonical
from locaii.tools import LocalTools


def test_update_during_prediction_defers_stale_write(task):
    """A newly arrived correction prevents an undispatched stale write."""
    store, identifier, workspace = task

    def generate_old_plan(messages, kwargs):
        store.queue_steering(identifier, "Do not write anything. Reply UPDATED.")
        return answer(
            "",
            [call("write_file", {"path": "stale.txt", "content": "old"}, "old")],
            "tool_calls",
        )

    def revised_answer(messages, kwargs):
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Do not write anything. Reply UPDATED."
        receipts = [m for m in messages if m["role"] == "tool"]
        assert json.loads(receipts[-1]["content"])["not_executed"] is True
        return answer("UPDATED")

    models = FakeModels([generate_old_plan, revised_answer])
    result = run(Engine(store, models), identifier, "Write the original output.")
    assert result["status"] == "completed" and result["text"] == "UPDATED"
    assert not (workspace / "stale.txt").exists()
    assert store.pending_steering(identifier) == []
    assert (
        len([e for e in store.events(identifier) if e["type"] == "steering_applied"])
        == 1
    )


def test_update_during_tool_preserves_done_effect_and_skips_next(task, monkeypatch):
    """An in-flight write completes once; the remaining old batch is reconsidered."""
    store, identifier, workspace = task
    original = LocalTools.execute
    calls = []

    def execute(tools, name, args):
        calls.append(name)
        result = original(tools, name, args)
        if name == "write_file":
            store.queue_steering(
                identifier, "Keep first.txt. Do not create second.txt. Reply REVISED."
            )
        return result

    monkeypatch.setattr(LocalTools, "execute", execute)

    def evaluate(messages, kwargs):
        assert "Keep first.txt" in messages[-1]["content"]
        receipts = [m for m in messages if m["role"] == "tool"]
        assert len(receipts) == 2
        assert json.loads(receipts[0]["content"])["success"]
        assert json.loads(receipts[1]["content"])["not_executed"]
        return answer("REVISED")

    models = FakeModels(
        [
            answer(
                "",
                [
                    call(
                        "write_file", {"path": "first.txt", "content": "kept"}, "first"
                    ),
                    call(
                        "write_file",
                        {"path": "second.txt", "content": "stale"},
                        "second",
                    ),
                ],
                "tool_calls",
            ),
            evaluate,
        ]
    )
    result = run(Engine(store, models), identifier, "Create two synthetic outputs.")
    assert result["status"] == "completed"
    assert (workspace / "first.txt").read_text() == "kept"
    assert not (workspace / "second.txt").exists()
    assert calls == ["write_file"]
    persisted = store.messages(identifier)
    index = next(i for i, m in enumerate(persisted) if m.get("_steering_id"))
    assert persisted[index - 1]["role"] == "tool"


def test_updates_after_final_prediction_are_ordered_and_answered(task):
    """A reply computed on the old context does not finish over newer instructions."""
    store, identifier, _ = task

    def old_answer(messages, kwargs):
        store.queue_steering(identifier, "Use French.")
        store.queue_steering(
            identifier, "Keep the original goal; answer in one sentence."
        )
        return answer("Old English answer.")

    def corrected(messages, kwargs):
        assert [m["content"] for m in messages[-2:]] == [
            "Use French.",
            "Keep the original goal; answer in one sentence.",
        ]
        assert "Original objective" in canonical(messages)
        return answer("Réponse corrigée.")

    result = run(
        Engine(store, FakeModels([old_answer, corrected])),
        identifier,
        "Original objective",
    )
    assert result["iterations"] == 2 and result["text"] == "Réponse corrigée."
    assert len([m for m in store.messages(identifier) if m.get("_steering_id")]) == 2


def test_stopped_updates_survive_restart_and_precede_next_user_turn(task):
    """Stopped work retains updates; subsequent user instructions remain newest."""
    store, identifier, _ = task
    cancel = threading.Event()

    def stopping(messages, kwargs):
        store.queue_steering(identifier, "Pending constraint")
        cancel.set()
        raise InterruptedError("Stopped fixture")

    result = run(
        Engine(store, FakeModels([stopping])),
        identifier,
        "Original task",
        cancel=cancel,
    )
    assert result["status"] == "interrupted"
    assert len(store.pending_steering(identifier)) == 1
    root = store.root
    store.close()
    reopened = Store(root)

    def resumed(messages, kwargs):
        users = [m["content"] for m in messages if m["role"] == "user"]
        assert users[-2:] == [
            "Pending constraint",
            "Continue with this additional requirement",
        ]
        return answer("Resumed with both constraints.")

    result = run(
        Engine(reopened, FakeModels([resumed])),
        identifier,
        "Continue with this additional requirement",
    )
    assert result["status"] == "completed" and not reopened.pending_steering(identifier)
    assert len([m for m in reopened.messages(identifier) if m.get("_steering_id")]) == 1


def test_update_during_final_memory_gets_another_iteration(task):
    """Memory inference cannot swallow a correction arriving just before completion."""
    store, identifier, workspace = task
    (workspace / "input.txt").write_text("fixture")
    models = FakeModels(
        [
            answer(
                "",
                [call("read_text_file", {"path": "input.txt"}, "read")],
                "tool_calls",
            ),
            answer("Old final"),
            answer("NEW FINAL"),
        ]
    )
    engine = Engine(store, models)
    sent = []

    def memory(*args):
        if not sent:
            sent.append(store.queue_steering(identifier, "Actually reply NEW FINAL."))

    engine._memory = memory
    result = run(engine, identifier, "Read input.txt and answer.")
    assert result["status"] == "completed" and result["text"] == "NEW FINAL"
    assert store.pending_steering(identifier) == []


def test_update_during_compaction_is_budgeted_before_prediction(task):
    """An update arriving during context work is included without a stale prediction."""
    store, identifier, _ = task

    def inspect(messages, kwargs):
        assert messages[-1]["content"] == "Apply this before predicting."
        return answer("Updated first prediction.")

    models = FakeModels([inspect])
    engine = Engine(store, models)
    compactions = []

    def compact(session_id, history, *args):
        compactions.append(canonical(history))
        if len(compactions) == 1:
            store.queue_steering(session_id, "Apply this before predicting.")
        return history, 100

    engine._compact = compact
    result = run(engine, identifier, "Original objective")
    assert result["iterations"] == 1
    assert len(compactions) == 2
    assert "Apply this before predicting." in compactions[1]
