"""User-selected attachments grant exact read access without broadening writes."""

import threading
from locaii.storage import Store
from locaii.tools import LocalTools


def test_approved_file_read_does_not_expand_workspace(tmp_path):
    store = Store(tmp_path / "state")
    task = store.create_session("Test", tmp_path / "workspace")
    outside = tmp_path / "chosen.txt"
    outside.write_text("selected evidence")
    other = tmp_path / "other.txt"
    other.write_text("not selected")
    tools = LocalTools(
        store,
        task["id"],
        {"read_all_files": False, "_approved_read_paths": [str(outside)]},
        threading.Event(),
        lambda *_: None,
    )
    assert (
        tools.execute("read_text_file", {"path": str(outside)})["content"]
        == "selected evidence"
    )
    assert "error" in tools.execute("read_text_file", {"path": str(other)})
    assert "error" in tools.execute(
        "write_file", {"path": str(outside), "content": "replacement"}
    )
    assert outside.read_text() == "selected evidence"


def test_binary_attachment_is_not_decoded_as_fake_text(tmp_path):
    store = Store(tmp_path / "state")
    task = store.create_session("Test", tmp_path / "workspace")
    binary = tmp_path / "workspace" / "binary.dat"
    binary.write_bytes(b"\x00\x00\xff\xfe")
    tools = LocalTools(store, task["id"], {}, threading.Event(), lambda *_: None)
    assert tools.read(str(binary))["binary"] is True


def test_broken_attachment_retains_original_request(tmp_path):
    import pytest
    from locaii.engine import Engine

    store = Store(tmp_path / "state")
    task = store.create_session("Test", tmp_path / "workspace")
    with pytest.raises(FileNotFoundError):
        Engine(store, object()).run(
            task["id"],
            "Inspect this attachment.",
            attachments=[str(tmp_path / "missing.txt")],
        )
    assert store.messages(task["id"])[0]["content"] == "Inspect this attachment."
    assert store.get_session(task["id"])["status"] == "error"
