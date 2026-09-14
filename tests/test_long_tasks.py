"""Regressions for the failure modes that ended long tasks prematurely.

Each test reproduces one observed cause from the recorded 2026-09-14 session:
an unusable shell boundary, a hard abort after a short run of tool failures,
compaction discarding an already-completed directory survey, and duplicated
receipts re-filling the context that compaction had just freed.
"""

from __future__ import annotations
import json
import threading
from pathlib import Path
import pytest
from locaii.engine import Engine, collapse_receipts, observation_ledger
from locaii.storage import Store, canonical
from locaii.tools import LocalTools
from test_engine import FakeModels, answer, call, run


@pytest.fixture
def task(tmp_path):
    """A workspace plus a separate readable data volume, as on the real machine."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "volume" / "dataset"
    outside.mkdir(parents=True)
    (outside / "metadata.csv").write_text("animal_id,session\n29538,baseline\n")
    store = Store(tmp_path / "state")
    session = store.create_session("Long task", workspace)
    return store, session["id"], workspace, outside


def tools(store, identifier, **options):
    settings = {
        "workspace_writes": True,
        "allow_terminal": True,
        "read_all_files": True,
        **options,
    }
    return LocalTools(
        store, identifier, settings, threading.Event(), lambda *args: None
    )


def test_readable_directory_is_usable_as_a_command_working_directory(task):
    """Reads and commands must share one boundary, or the model cannot proceed."""
    store, identifier, _workspace, outside = task
    local = tools(store, identifier)
    assert not local.execute("list_directory", {"path": str(outside)}).get("error")
    result = local.execute(
        "run_command",
        {"command": "cat metadata.csv", "cwd": str(outside), "timeout": 20},
    )
    assert result["status"] == "completed", result
    assert "29538" in result["stdout"]


def test_restricted_reads_still_refuse_outside_commands_and_name_the_boundary(task):
    store, identifier, workspace, outside = task
    local = tools(store, identifier, read_all_files=False)
    result = local.execute(
        "run_command", {"command": "cat metadata.csv", "cwd": str(outside)}
    )
    assert result["success"] is False
    # The refusal must state the actual workspace and stop path guessing.
    assert str(workspace) in result["error"]
    assert "variants" in result["error"]


def test_missing_working_directory_reports_where_it_resolved(task):
    store, identifier, workspace, _outside = task
    local = tools(store, identifier)
    result = local.execute("run_command", {"command": "ls", "cwd": "./29538"})
    assert result["success"] is False
    assert str(workspace / "29538") in result["error"]
    assert str(workspace) in result["error"]


def test_three_tool_failures_are_coached_instead_of_ending_the_task(task):
    """A short run of bad guesses must not discard the completed work."""
    store, identifier, _workspace, _outside = task
    missing = [
        answer(
            "",
            [call("list_directory", {"path": f"missing-{i}"}, f"m{i}")],
            "tool_calls",
        )
        for i in range(3)
    ]
    models = FakeModels(missing + [answer("The folder is outside the workspace.")])
    result = run(Engine(store, models), identifier, "Survey the dataset.")
    assert result["status"] == "completed"
    guidance = [
        m
        for prompt in models.prompts
        for m in prompt
        if "three tool calls failed in a row" in str(m.get("content", ""))
    ]
    assert guidance, "corrective guidance was never delivered"
    assert any(event["type"] == "recovery" for event in store.events(identifier))


def test_persistent_failures_stop_only_after_two_corrections(task):
    store, identifier, _workspace, _outside = task
    missing = [
        answer(
            "",
            [call("list_directory", {"path": f"missing-{i}"}, f"m{i}")],
            "tool_calls",
        )
        for i in range(9)
    ]
    models = FakeModels(missing)
    with pytest.raises(RuntimeError, match="two corrective attempts"):
        run(Engine(store, models), identifier, "Survey the dataset.")
    recoveries = [e for e in store.events(identifier) if e["type"] == "recovery"]
    assert len(recoveries) == 2


def test_ledger_preserves_a_completed_survey_across_compaction(task):
    """The directory map survives at a fraction of the original token cost."""
    store, identifier, _workspace, _outside = task
    store.append(identifier, "user", "Describe the organization of the dataset.")
    store.append(
        identifier,
        answer("", [call("list_directory", {"path": "/volume/dataset"}, "survey")])[
            "message"
        ],
    )
    listing = {
        "path": "/volume/dataset",
        "entries": [
            {"name": f"animal-{index}", "directory": True, "bytes": None}
            for index in range(30)
        ],
        "padding": "verbose receipt " * 4000,
    }
    store.append(identifier, "tool", canonical(listing), tool_call_id="survey")
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
    text = canonical(compacted)
    assert "animal-0" in text and "animal-29" in text
    assert "do not repeat them" in text


def test_duplicate_receipts_are_collapsed_to_one_copy():
    """Identical repeated observations must not be paid for three times."""
    body = canonical({"path": "/volume/dataset", "entries": ["a"] * 500})
    history = []
    for index in range(3):
        history.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call("list_directory", {"path": "/volume/dataset"}, f"c{index}")
                ],
            }
        )
        history.append(
            {
                "role": "tool",
                "name": "list_directory",
                "tool_call_id": f"c{index}",
                "content": body,
            }
        )
    before = len(canonical(history))
    collapsed = collapse_receipts([dict(m) for m in history])
    after = len(canonical(collapsed))
    assert after < before / 2
    assert sum(m.get("content") == body for m in collapsed) == 1
    assert all(m.get("tool_call_id") for m in collapsed if m["role"] == "tool")


def test_larger_page_supersedes_a_smaller_read_of_the_same_range():
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                call("read_text_file", {"path": "notes.md", "offset_chars": 0}, "small")
            ],
        },
        {"role": "tool", "tool_call_id": "small", "content": '{"content":"abc"}'},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                call("read_text_file", {"path": "notes.md", "offset_chars": 0}, "large")
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "large",
            "content": '{"content":"abcdefghij"}',
        },
    ]
    collapsed = collapse_receipts(history)
    assert json.loads(collapsed[1]["content"])["superseded"] is True
    assert "abcdefghij" in collapsed[3]["content"]


def test_ledger_skips_failures_and_records_outcomes():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("list_directory", {"path": "/data"}, "ok")],
        },
        {
            "role": "tool",
            "tool_call_id": "ok",
            "content": canonical(
                {"path": "/data", "entries": [{"name": "baseline", "directory": True}]}
            ),
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("list_directory", {"path": "/nope"}, "bad")],
        },
        {
            "role": "tool",
            "tool_call_id": "bad",
            "content": canonical({"success": False, "error": "missing"}),
        },
    ]
    ledger = observation_ledger(messages)
    assert "listed /data -> 1 entries: baseline/" in ledger
    assert "/nope" not in ledger


def test_recorded_session_pattern_completes_end_to_end(task):
    """Replay of the real 2026-09-14 abort: outside data, then a shell step."""
    store, identifier, _workspace, outside = task
    script = [
        answer(
            "", [call("list_directory", {"path": str(outside)}, "l1")], "tool_calls"
        ),
        answer(
            "",
            [call("read_text_file", {"path": str(outside / "metadata.csv")}, "r1")],
            "tool_calls",
        ),
        answer(
            "",
            [
                call(
                    "run_command",
                    {"command": "head -1 metadata.csv", "cwd": str(outside)},
                    "c1",
                )
            ],
            "tool_calls",
        ),
        answer("The dataset is organized by animal and session."),
    ]
    models = FakeModels(script)
    result = run(
        Engine(store, models),
        identifier,
        "Explain the organization of " + str(outside),
        options={"allow_terminal": True, "read_all_files": True},
    )
    assert result["status"] == "completed"
    assert "unresolved_tool_failures" not in result
    receipts = [m for m in store.messages(identifier) if m["role"] == "tool"]
    assert all("outside the selected workspace" not in m["content"] for m in receipts)
