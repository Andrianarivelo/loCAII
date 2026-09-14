"""Persistent execution preserves user goals and completed effects across recovery."""

import copy
import json
import threading
import time
from pathlib import Path

import pytest

from locaii.engine import Engine
from locaii.models import ModelCancelled, ModelError
from test_engine import FakeModels, answer, call, run, task


def persistent_run(engine, identifier, text, **kwargs):
    """Exercise unlimited autonomous mode instead of the legacy bounded fixture."""
    return run(
        engine,
        identifier,
        text,
        options={"persistent_mode": True, "max_iterations": 0},
        **kwargs,
    )


@pytest.fixture
def recovery(monkeypatch):
    """Record recovery attempts without slowing every deterministic regression."""
    waits = []

    def wait(engine, session_id, cancel, emit, attempt, reason):
        """Retain the reason and cancellation contract without sleeping."""
        waits.append((attempt, reason))
        if cancel.is_set():
            raise InterruptedError("Stopped during recovery.")

    monkeypatch.setattr(Engine, "_recovery_wait", wait)
    return waits


def audits(monkeypatch, verdicts):
    """Record engine-to-review inputs while returning explicit scripted judgments."""
    recorded = []
    sequence = iter(verdicts)

    def audit(models, requests, observations, candidate, cancel, context=32768):
        """Inspect original goals and current evidence without a second inference."""
        recorded.append(
            copy.deepcopy(
                {
                    "requests": requests,
                    "observations": observations,
                    "candidate": candidate,
                }
            )
        )
        return {
            "verdict": next(sequence),
            "summary": "Compared with original requirements.",
            "next_step": "Complete and verify the remaining requirement.",
        }

    monkeypatch.setattr("locaii.engine.audit_completion", audit)
    return recorded


def test_partial_answer_continues_until_remaining_action_is_verified(
    task, monkeypatch, recovery
):
    """A fluent partial answer must not end the task before the remaining write."""
    store, identifier, workspace = task
    checks = audits(monkeypatch, ["continue", "complete"])
    models = FakeModels(
        [
            answer("I have prepared the first part; you can make result.txt next."),
            answer(
                "Writing the remaining result.",
                [
                    call(
                        "write_file",
                        {"path": "result.txt", "content": "verified"},
                        "write",
                    )
                ],
            ),
            answer("Both requested steps are complete; result.txt contains verified."),
        ]
    )
    events = []
    result = persistent_run(
        Engine(store, models),
        identifier,
        "Prepare the answer and create result.txt containing verified.",
        emit=lambda kind, data: events.append((kind, data)),
    )
    assert result["status"] == "completed"
    assert (workspace / "result.txt").read_text() == "verified"
    assert len(checks) == 2 and recovery
    assert "result.txt" in checks[-1]["observations"]
    assert sum(kind == "done" for kind, _ in events) == 1
    assert not any(kind == "error" for kind, _ in events)


def test_continue_audit_retains_initial_objective_and_later_constraints(
    task, monkeypatch, recovery
):
    """Continue is not allowed to replace the earlier task or its corrections."""
    store, identifier, _ = task
    original = "Explain folder organization and verify every session."
    correction = "Do not change metadata, and include hidden session folders."
    store.append(identifier, "user", original)
    store.append(identifier, "assistant", "A partial explanation.")
    store.append(identifier, "user", correction)
    checks = audits(monkeypatch, ["complete"])
    result = persistent_run(
        Engine(store, FakeModels([answer("Verified the requested structure.")])),
        identifier,
        "Continue",
    )
    assert result["status"] == "completed"
    assert [item["content"] for item in checks[0]["requests"]] == [
        original,
        correction,
        "Continue",
    ]


def test_model_start_retries_without_duplicating_user_request(
    task, monkeypatch, recovery
):
    """An unavailable local runtime should retry before consuming the saved goal twice."""
    store, identifier, _ = task
    audits(monkeypatch, ["complete"])
    models = FakeModels([answer("The answer is four.")])
    attempts = []

    def start(model_id, **kwargs):
        """Fail startup twice to exercise the dedicated model-operation retry path."""
        attempts.append(model_id)
        if len(attempts) < 3:
            raise ModelError("Temporary local startup failure")

    models.start = start
    result = persistent_run(Engine(store, models), identifier, "What is two plus two?")
    assert result["status"] == "completed"
    # The main loop may also ensure the successfully started model is still
    # loaded. Only the two failed attempts should trigger recovery delays.
    assert len(attempts) >= 3 and len(recovery) == 2
    requests = [
        m
        for m in store.messages(identifier)
        if m["role"] == "user" and not m.get("_runtime")
    ]
    assert [m["content"] for m in requests] == ["What is two plus two?"]


def test_stream_retry_keeps_completed_write_receipt(task, monkeypatch, recovery):
    """A stream failure and repeated write request must not replay a completed effect."""
    store, identifier, workspace = task
    audits(monkeypatch, ["complete"])
    request = {"path": "result.txt", "content": "single effect"}
    timestamp = []

    def remember_then_fail(messages, kwargs):
        """Capture the actual write timestamp before the backend connection fails."""
        timestamp.append((workspace / "result.txt").stat().st_mtime_ns)
        raise ModelError("Inference stream ended unexpectedly")

    models = FakeModels(
        [
            answer("", [call("write_file", request, "initial")]),
            remember_then_fail,
            answer("", [call("write_file", request, "retry")]),
            answer("The file is verified."),
        ]
    )
    result = persistent_run(
        Engine(store, models), identifier, "Write result.txt once and verify it."
    )
    assert result["status"] == "completed" and recovery
    assert (workspace / "result.txt").stat().st_mtime_ns == timestamp[0]
    receipts = [
        json.loads(m["content"])
        for m in store.messages(identifier)
        if m["role"] == "tool"
    ]
    assert receipts[-1]["reused_receipt"] is True
    assert (
        sum(
            m["role"] == "user" and not m.get("_runtime")
            for m in store.messages(identifier)
        )
        == 1
    )


def test_steering_during_completion_audit_invalidates_old_verdict(
    task, monkeypatch, recovery
):
    """An audit that passed before a new user requirement cannot finish the updated task."""
    store, identifier, workspace = task
    checks = []

    def audit(models, requests, observations, candidate, cancel, context=32768):
        """Deliver a new requirement while the first local review is in flight."""
        checks.append(copy.deepcopy(requests))
        if len(checks) == 1:
            store.queue_steering(
                identifier, "Also create verification.txt containing checked."
            )
        return {
            "verdict": "complete",
            "summary": "Current request satisfied.",
            "next_step": "",
        }

    monkeypatch.setattr("locaii.engine.audit_completion", audit)
    models = FakeModels(
        [
            answer("Original explanation finished."),
            answer(
                "Applying the additional instruction.",
                [
                    call(
                        "write_file",
                        {"path": "verification.txt", "content": "checked"},
                        "steered",
                    )
                ],
            ),
            answer("The explanation and additional file are complete."),
        ]
    )
    result = persistent_run(Engine(store, models), identifier, "Explain the project.")
    assert result["status"] == "completed"
    assert len(checks) == 2
    assert (workspace / "verification.txt").read_text() == "checked"
    assert (
        checks[-1][-1]["content"] == "Also create verification.txt containing checked."
    )


@pytest.mark.parametrize("stage", ["start", "stream", "audit"])
def test_model_cancelled_stops_persistent_execution(task, monkeypatch, recovery, stage):
    """ModelCancelled is never reclassified as a retryable inference error."""
    store, identifier, _ = task
    models = FakeModels([answer("Potential final answer.")])

    def cancelled(*args, **kwargs):
        """Simulate cancellation at each model inference boundary."""
        raise ModelCancelled("Stopped by user")

    if stage == "start":
        models.start = cancelled
    elif stage == "stream":
        models.responses = [ModelCancelled("Stopped by user")]
    else:
        monkeypatch.setattr("locaii.engine.audit_completion", cancelled)
    result = persistent_run(Engine(store, models), identifier, "Complete this task.")
    assert result["status"] == "interrupted"
    assert not recovery


def test_empty_csv_recovers_to_an_available_valid_input(task, monkeypatch, recovery):
    """Verified missing rows must not stop independent work on a valid alternate input."""
    store, identifier, workspace = task
    (workspace / "empty.csv").write_text("left,right\n")
    (workspace / "valid.csv").write_text("left,right\n10,20\n")
    audits(monkeypatch, ["complete"])
    models = FakeModels(
        [
            *[
                answer(
                    "", [call("read_text_file", {"path": "empty.csv"}, f"empty-{i}")]
                )
                for i in range(8)
            ],
            answer(
                "Checking the alternate table.",
                [
                    call(
                        "inspect_csv",
                        {"path": "valid.csv", "column_names": ["left", "right"]},
                        "valid",
                    )
                ],
            ),
            answer(
                "The alternate table has left=10 and right=20; the initial table has no rows."
            ),
        ]
    )
    events = []
    result = persistent_run(
        Engine(store, models),
        identifier,
        "Compare left and right using an available nonempty table.",
        emit=lambda kind, data: events.append((kind, data)),
    )
    assert result["status"] == "completed" and recovery
    assert "left=10" in result["text"]
    assert not any(kind in {"input_required", "error"} for kind, _ in events)
    assert (workspace / "empty.csv").read_text() == "left,right\n"


def test_real_backoff_reacts_to_stop_in_under_half_a_second(task):
    """Even the maximum sixty-second recovery delay must wake promptly on Stop."""
    store, identifier, _ = task
    cancel = threading.Event()
    engine = Engine(store, FakeModels())
    timer = threading.Timer(0.03, cancel.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(InterruptedError, match="stopped"):
            engine._recovery_wait(
                identifier, cancel, lambda *_: None, 20, "temporary failure"
            )
    finally:
        timer.cancel()
        timer.join()
    assert time.monotonic() - started < 0.5


def test_audit_model_error_retries_without_repeating_task_inference(
    task, monkeypatch, recovery
):
    """A failed verification request retries the review, preserving the candidate."""
    store, identifier, _ = task
    attempts = []

    def audit(models, requests, observations, candidate, cancel, context=32768):
        """Fail only the first review request, then accept the same verified answer."""
        attempts.append(candidate)
        if len(attempts) == 1:
            raise ModelError("Temporary review inference failure")
        return {
            "verdict": "complete",
            "summary": "The answer is sufficient.",
            "next_step": "",
        }

    monkeypatch.setattr("locaii.engine.audit_completion", audit)
    models = FakeModels([answer("Four.")])
    result = persistent_run(Engine(store, models), identifier, "What is two plus two?")
    assert result["status"] == "completed"
    assert attempts == ["Four.", "Four."]
    assert len(models.prompts) == 1 and len(recovery) == 1


def test_waiting_review_remains_active_and_rechecks_progress(
    task, monkeypatch, recovery
):
    """A review's waiting verdict must not require the user to press Continue."""
    store, identifier, _ = task
    checks = audits(monkeypatch, ["waiting", "complete"])
    models = FakeModels(
        [
            answer("The required external input is not yet available."),
            answer(
                "The prerequisite is now available and the complete request is verified."
            ),
        ]
    )
    result = persistent_run(
        Engine(store, models),
        identifier,
        "Wait for the prerequisite and verify the entire request.",
    )
    assert result["status"] == "completed"
    assert len(checks) == 2 and len(recovery) == 1


def test_three_hundred_pages_survive_multiple_real_context_compactions(
    task, monkeypatch, recovery
):
    """Simulate long task control flow, without claiming hours of real-model reasoning."""
    store, identifier, workspace = task
    initial = "Inspect every page of pages.txt in sequence and verify all 300 pages."
    constraints = (
        "Do not change pages.txt, and report that every requested page was inspected."
    )
    store.append(identifier, "user", initial)
    store.append(identifier, "user", constraints)
    content = "".join(
        f"page {index:03d}: ".ljust(255, "x") + "\n" for index in range(300)
    )
    source = workspace / "pages.txt"
    source.write_text(content)
    checks = audits(monkeypatch, ["complete"])
    models = FakeModels(
        [
            *[
                answer(
                    "",
                    [
                        call(
                            "read_text_file",
                            {
                                "path": "pages.txt",
                                "offset_chars": index * 256,
                                "max_chars": 256,
                            },
                            f"page-{index}",
                        )
                    ],
                )
                for index in range(300)
            ],
            answer(
                "Every requested page was inspected: all 300 pages, with the source unchanged."
            ),
        ],
        context=12000,
    )
    events = []
    result = persistent_run(
        Engine(store, models),
        identifier,
        "Continue",
        emit=lambda kind, data: events.append((kind, data)),
    )
    assert result["status"] == "completed" and result["tool_rounds"] == 300
    assert len(checks) == 1
    assert [item["content"] for item in checks[0]["requests"]] == [
        initial,
        constraints,
        "Continue",
    ]
    assert source.read_text() == content
    compactions = [data for kind, data in events if kind == "compaction"]
    assert len(compactions) >= 2
    assert all(item["after_tokens"] < item["before_tokens"] for item in compactions)
    assert all(Path(item["archive"]).is_file() for item in compactions)
    checkpoint = store.checkpoint(identifier)
    assert checkpoint["covered_count"] > 300
    assert Path(checkpoint["archive"]).is_file()
    recorded = store.events(identifier, limit=1000)
    assert any(event["type"] == "compaction" for event in recorded)
    assert not any(kind in {"input_required", "error"} for kind, _ in events)
    assert not recovery
    reads = [
        json.loads(message["content"])
        for message in store.messages(identifier)
        if message["role"] == "tool"
    ]
    assert [receipt["offset_chars"] for receipt in reads] == [
        index * 256 for index in range(300)
    ]
    assert all(
        initial in str(prompt) and constraints in str(prompt)
        for prompt in models.prompts
        if not prompt[0]["content"].startswith("Write concise Markdown task memory")
    )


def test_pending_instruction_wakes_backoff_only_once(task):
    """A backend outage with undelivered steering must not spin on the same update."""
    store, identifier, _ = task
    engine = Engine(store, FakeModels())
    store.queue_steering(identifier, "Use the latest local input.")
    cancel = threading.Event()
    engine._recovery_wait(identifier, cancel, lambda *_: None, 8, "backend offline")
    # The same queued message cannot skip the next wait. Cancellation still can.
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(InterruptedError):
            engine._recovery_wait(
                identifier, cancel, lambda *_: None, 8, "backend offline"
            )
        assert time.monotonic() - started < 0.5
    finally:
        timer.cancel()


def test_large_history_does_not_hit_old_archive_cutoff(task):
    """A real archive beyond64MiB preserves its contents instead of ending long work."""
    store, identifier, _ = task
    text = "x" * (64 * 1024 * 1024)
    messages = [{"role": "tool", "name": "read_text_file", "content": text}]
    path = store.archive(identifier, messages)
    assert path.stat().st_size > 64 * 1024 * 1024
    assert json.loads(path.read_text())["messages"] == messages
    path.unlink()  # Do not retain the large disposable fixture after verification.
