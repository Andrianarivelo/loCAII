"""Filesystem boundaries and genuine command receipts on synthetic files only."""

from pathlib import Path
import json
import threading
import time
import pytest
from locaii.storage import Store
from locaii.tools import LocalTools


@pytest.fixture
def tools(tmp_path):
    store = Store(tmp_path / "state")
    task = store.create_session("Synthetic tools test", tmp_path / "workspace")
    return LocalTools(
        store,
        task["id"],
        {"workspace_writes": True, "read_all_files": False, "allow_terminal": False},
        threading.Event(),
        lambda *args: None,
    )


def test_native_writes_preserve_existing_raw_and_metadata(tools):
    for name in ["original.csv", "animal_metadata.txt", "recording.nwb"]:
        path = tools.workspace / name
        path.write_text("ORIGINAL")
        result = tools.execute("write_file", {"path": name, "content": "changed"})
        assert result["success"] is False
        assert "protected" in result["error"]
        assert path.read_text() == "ORIGINAL"
    derived = tools.execute(
        "write_file", {"path": "new_derived.csv", "content": "derived"}
    )
    assert derived["success"] is True


def test_native_writes_resolve_symlink_before_boundary(tools, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    original = outside / "note.txt"
    original.write_text("outside")
    (tools.workspace / "link").symlink_to(outside, target_is_directory=True)
    result = tools.execute(
        "write_file", {"path": "link/note.txt", "content": "changed"}
    )
    assert result["success"] is False
    assert original.read_text() == "outside"


def test_edit_requires_exact_unique_match_and_preserves_backup(tools):
    path = tools.workspace / "notes.md"
    path.write_text("A B A")
    denied = tools.execute(
        "edit_file", {"path": "notes.md", "old_text": "A", "new_text": "Z"}
    )
    assert denied["success"] is False
    result = tools.execute(
        "edit_file", {"path": "notes.md", "old_text": "B", "new_text": "C"}
    )
    assert result["success"] is True
    assert path.read_text() == "A C A"
    assert Path(result["backup"]).read_text() == "A B A"


def test_utf8_read_pagination_is_character_based(tools):
    content = "é猫" * 700
    (tools.workspace / "unicode.txt").write_text(content)
    first = tools.execute("read_text_file", {"path": "unicode.txt", "max_chars": 256})
    second = tools.execute(
        "read_text_file",
        {
            "path": "unicode.txt",
            "offset_chars": first["next_offset_chars"],
            "max_chars": 256,
        },
    )
    assert first["content"] + second["content"] == content[:512]


def test_disabled_terminal_cannot_be_called_by_model(tools):
    result = tools.execute("run_command", {"command": "echo unreachable"})
    assert result["success"] is False
    assert "not enabled" in result["error"]


def test_failed_command_receipt_ignores_stdout_success_claim(tools):
    tools.options["allow_terminal"] = True
    result = tools.execute("run_command", {"command": "printf 'success=true'; exit 7"})
    assert result["status"] == "failed"
    assert result["exit_code"] == 7
    assert result["stdout"] == "success=true"


def test_durable_job_is_polled_until_actual_completion(tools):
    tools.options["allow_terminal"] = True
    result = tools.execute("start_command", {"command": "sleep 0.5; printf finished"})
    deadline = time.monotonic() + 5
    while result["status"] in {"starting", "running"} and time.monotonic() < deadline:
        time.sleep(0.1)
        result = tools.execute("poll_command", {"job_id": result["job_id"]})
    assert result["status"] == "completed"
    assert result["exit_code"] == 0
    assert result["stdout"] == "finished"


def test_job_ids_cannot_escape_task_directory(tools):
    tools.options["allow_terminal"] = True
    result = tools.execute("poll_command", {"job_id": "../../private"})
    assert result["success"] is False
    assert "Invalid job" in result["error"]


def test_repeated_unchanged_failure_is_bounded(tools):
    args = {"path": "does-not-exist.txt"}
    for _ in range(2):
        assert tools.execute("read_text_file", args)["success"] is False
    result = tools.execute("read_text_file", args)
    assert "failed twice" in result["error"]


def test_archive_read_is_private_and_paginated(tools, tmp_path):
    archive = tools.store.archive(
        tools.session_id, [{"role": "tool", "content": "a" * 9000}]
    )
    result = tools.execute(
        "read_context_archive", {"archive_path": str(archive), "message_index": 0}
    )
    assert result["historical_data_only"] is True
    assert result["next_offset_chars"] == 6000
    denied = tools.execute(
        "read_context_archive",
        {"archive_path": str(tmp_path / archive.name), "message_index": 0},
    )
    assert denied["success"] is False


def test_metadata_pauses_until_external_observation(tmp_path):
    store = Store(tmp_path / "state")
    session = store.create_session("Metadata", tmp_path / "work")
    tools = LocalTools(store, session["id"], {}, threading.Event(), lambda *_: None)
    assert tools.execute(
        "update_memory", {"markdown": "# Test memory\nNo external work completed."}
    )["saved"]
    assert "update_memory" not in {s["function"]["name"] for s in tools.schemas()}
    tools.execute("list_directory", {"path": str(tmp_path / "work")})
    assert "update_memory" in {s["function"]["name"] for s in tools.schemas()}


def test_snapshot_cache_invalidates_external_file_change(tmp_path):
    store = Store(tmp_path / "state")
    session = store.create_session("Snapshots", tmp_path / "work")
    tools = LocalTools(store, session["id"], {}, threading.Event(), lambda *_: None)
    path = tmp_path / "work" / "note.txt"
    path.write_text("FIRST")
    assert tools.execute("read_text_file", {"path": str(path)})["content"] == "FIRST"
    assert tools.execute("read_text_file", {"path": str(path)})["reused_snapshot"]
    path.write_text("SECOND")
    assert tools.execute("read_text_file", {"path": str(path)})["content"] == "SECOND"


def test_cancel_marker_stops_starting_durable_job(tmp_path):
    from locaii.storage import atomic_json
    import subprocess, sys

    store = Store(tmp_path / "state")
    session = store.create_session("Cancelled startup", tmp_path / "work")
    tools = LocalTools(
        store,
        session["id"],
        {"allow_terminal": True},
        threading.Event(),
        lambda *_: None,
    )
    identifier = "a" * 32
    path = tools._job_path(identifier)
    atomic_json(
        path,
        {
            "job_id": identifier,
            "status": "starting",
            "command": "touch should-not-exist",
            "cwd": str(tmp_path / "work"),
        },
    )
    tools.cancel_job(identifier)
    subprocess.run([sys.executable, "-m", "locaii.job_worker", str(path)], check=True)
    assert tools.poll(identifier)["status"] == "cancelled"
    assert not (tmp_path / "work" / "should-not-exist").exists()


def test_orphaned_startup_is_unknown_and_active_jobs_are_not_hidden(tools):
    from locaii.storage import atomic_json

    # More than 30 historical receipts must not hide an older running command.
    for index in range(35):
        identifier = f"{index:032x}"
        atomic_json(
            tools._job_path(identifier),
            {
                "job_id": identifier,
                "status": "completed",
                "created": time.time() + index,
            },
        )
    identifier = "f" * 32
    atomic_json(
        tools._job_path(identifier),
        {"job_id": identifier, "status": "starting", "created": time.time() - 60},
    )
    assert tools.poll(identifier)["status"] == "outcome_unknown"
    import os

    identifier = "e" * 32
    atomic_json(
        tools._job_path(identifier),
        {"job_id": identifier, "status": "running", "pid": os.getpid(), "created": 0},
    )
    assert any(
        job["job_id"] == identifier and job["status"] == "running"
        for job in tools.jobs()
    )
