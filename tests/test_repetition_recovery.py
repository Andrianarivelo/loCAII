"""Synthetic regressions for continuous work and evidence-based loop recovery."""

import threading
import pytest
from locaii.engine import Engine
from locaii.tools import LocalTools
from test_engine import FakeModels, answer, call, run, task


def reads(count, path="data.txt"):
    """Build deterministic identical read requests with unique protocol IDs."""
    return [
        answer("Reading the data.", [call("read_text_file", {"path": path}, f"r{i}")])
        for i in range(count)
    ]


def test_recovery_points_to_next_page_and_preserves_findings(task):
    store, identifier, workspace = task
    (workspace / "data.txt").write_text("a" * 6000 + "SECOND_PAGE")
    models = FakeModels(
        reads(3)
        + [
            answer(
                "The first page is known; I will inspect the next page.",
                [
                    call(
                        "read_text_file",
                        {"path": "data.txt", "offset_chars": 6000},
                        "next",
                    )
                ],
            ),
            answer("The next page contains SECOND_PAGE."),
        ]
    )
    result = run(Engine(store, models), identifier, "Inspect both pages")
    guidance = [
        m["content"]
        for m in models.prompts[3]
        if "Application recovery: repeated" in str(m.get("content"))
    ]
    assert guidance and '"next_offset_chars":6000' in guidance[0]
    assert "Explain briefly" in guidance[0]
    assert result["status"] == "completed"
    assert len([e for e in store.events(identifier) if e["type"] == "recovery"]) == 1


def test_persistent_read_loop_reports_incomplete_and_stops(task):
    store, identifier, workspace = task
    (workspace / "data.txt").write_text("evidence")
    models = FakeModels(reads(9))
    with pytest.raises(RuntimeError, match="No progress.*task is incomplete"):
        run(Engine(store, models), identifier, "Inspect", options={"max_iterations": 0})
    assert len(models.responses) == 2  # Initial read + two repeats per recovery cycle.
    assert store.get_session(identifier)["status"] == "error"
    assert not any(e["type"] == "done" for e in store.events(identifier))


@pytest.mark.parametrize("limit", [None, 0])
def test_default_continues_past_previous_forty_steps(task, limit):
    store, identifier, workspace = task
    (workspace / "data.txt").write_text("a" * (45 * 256))
    models = FakeModels(
        [
            answer(
                "",
                [
                    call(
                        "read_text_file",
                        {"path": "data.txt", "offset_chars": i * 256, "max_chars": 256},
                        f"page{i}",
                    )
                ],
            )
            for i in range(45)
        ]
        + [answer("All pages inspected.")]
    )
    options = {"model_id": "fixture"}
    if limit is not None:
        options["max_iterations"] = limit
    result = Engine(store, models).run(
        identifier, "Inspect each page", options={**options, "persistent_mode": False}
    )
    assert result["iterations"] == 46


def test_external_file_changes_count_as_new_evidence(task):
    store, identifier, workspace = task
    path = workspace / "data.txt"
    path.write_text("first")

    def changed(_messages, _options):
        path.write_text(path.read_text() + " changed")
        return reads(1)[0]

    models = FakeModels(reads(1) + [changed] * 8 + [answer("Observed changes.")])
    assert (
        run(Engine(store, models), identifier, "Watch updates")["status"] == "completed"
    )
    assert not any(e["type"] == "recovery" for e in store.events(identifier))


def test_repeated_completed_commands_are_not_reexecuted(task, monkeypatch):
    store, identifier, _workspace = task
    executions = []
    original = LocalTools.execute

    def execute(self, name, args):
        if name == "run_command":
            executions.append(args)
            return {"status": "completed", "exit_code": 0, "stdout": "observed"}
        return original(self, name, args)

    monkeypatch.setattr(LocalTools, "execute", execute)
    models = FakeModels(
        [
            answer("", [call("run_command", {"command": "test"}, f"c{i}")])
            for i in range(7)
        ]
    )
    with pytest.raises(RuntimeError, match="No progress"):
        run(
            Engine(store, models),
            identifier,
            "Inspect",
            options={"allow_terminal": True},
        )
    assert len(executions) == 1


def test_unchanged_polling_is_exempt_and_stop_is_honored(task, monkeypatch):
    store, identifier, _workspace = task
    cancel = threading.Event()
    polls = []
    original = LocalTools.execute

    def execute(self, name, args):
        if name == "poll_command":
            polls.append(args)
            if len(polls) == 8:
                cancel.set()
            return {"job_id": "fixture", "status": "running", "stdout": ""}
        return original(self, name, args)

    monkeypatch.setattr(LocalTools, "execute", execute)
    models = FakeModels(
        [
            answer("", [call("poll_command", {"job_id": "fixture"}, f"p{i}")])
            for i in range(9)
        ]
    )
    result = run(
        Engine(store, models),
        identifier,
        "Wait for the job",
        cancel=cancel,
        options={"max_iterations": 0},
    )
    assert result["status"] == "interrupted"
    assert len(polls) == 8
    assert not any(e["type"] == "recovery" for e in store.events(identifier))


def test_alternating_old_reads_are_still_no_progress(task):
    store, identifier, workspace = task
    for name in ("a.txt", "b.txt"):
        (workspace / name).write_text(name)
    models = FakeModels(
        [
            answer(
                "",
                [
                    call(
                        "read_text_file",
                        {"path": "a.txt" if i % 2 == 0 else "b.txt"},
                        f"r{i}",
                    )
                ],
            )
            for i in range(8)
        ]
    )
    with pytest.raises(RuntimeError, match="No progress"):
        run(Engine(store, models), identifier, "Inspect", options={"max_iterations": 0})


def test_fresh_evidence_in_mixed_batch_prevents_false_alarm(task):
    store, identifier, workspace = task
    (workspace / "data.txt").write_text("a" * 4000)
    models = FakeModels(
        [
            answer(
                "",
                [
                    call(
                        "read_text_file",
                        {"path": "data.txt", "max_chars": 256},
                        f"old{i}",
                    ),
                    call(
                        "read_text_file",
                        {
                            "path": "data.txt",
                            "max_chars": 256,
                            "offset_chars": (i + 1) * 256,
                        },
                        f"new{i}",
                    ),
                ],
            )
            for i in range(8)
        ]
        + [answer("Inspected the pages.")]
    )
    assert run(Engine(store, models), identifier, "Inspect")["status"] == "completed"
    assert not any(e["type"] == "recovery" for e in store.events(identifier))
