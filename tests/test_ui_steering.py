"""Synthetic worker tests for durable live chat steering, without model providers."""

import threading
import time

import pytest
from PySide6.QtTest import QTest

from test_ui import FakeStore, create_window


class SteeringStore(FakeStore):
    """Thread-safe in-memory steering receipts model the storage contract."""

    def __init__(self, root):
        super().__init__(root)
        self.updates = []
        self.lock = threading.Lock()

    def queue_steering(self, session_id, text):
        with self.lock:
            item = {
                "id": str(len(self.updates) + 1),
                "text": text,
                "session_id": session_id,
                "status": "pending",
            }
            self.updates.append(item)
            return dict(item)

    def pending_steering(self, session_id):
        with self.lock:
            return [
                dict(item)
                for item in self.updates
                if item["session_id"] == session_id and item["status"] == "pending"
            ]

    def consume(self, session_id):
        with self.lock:
            for item in self.updates:
                if item["session_id"] == session_id:
                    item["status"] = "applied"


def setup_window(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.store = SteeringStore(tmp_path)
    return app, window, engine


def spin(app, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert condition()
    app.processEvents()


def begin_held_chat(app, window, engine, *, status="completed", consume=True):
    """Keep a deterministic worker alive while the UI receives a new instruction."""
    entered, release = threading.Event(), threading.Event()

    def run(session_id, text, attachments, options, cancel, emit):
        engine.calls.append((session_id, text, attachments, options))
        emit("delta", {"text": "Original response."})
        entered.set()
        while not release.wait(0.005):
            if cancel.is_set():
                return {"status": "interrupted", "content": "Original response."}
        if consume:
            pending = window.store.pending_steering(session_id)
            window.store.consume(session_id)
            if pending:
                emit("steering_applied", {"ids": [item["id"] for item in pending]})
                emit("delta", {"text": "Updated response."})
                return {"status": status, "content": "Updated response."}
        return {"status": status, "content": "Original response."}

    engine.run = run
    window.composer.setPlainText("Original task")
    window.send()
    spin(app, lambda: entered.is_set() and window.stream_message is not None)
    return release


def test_live_update_is_queued_without_interrupting_and_streams_after_user(tmp_path):
    app, window, engine = setup_window(tmp_path)
    release = begin_held_chat(app, window, engine)
    worker, initial = window.worker, window.stream_message
    assert window.send_button.isEnabled()
    assert window.send_button.text() == "Send update"
    window.composer.setPlainText("Please use a shorter explanation")
    window.send()
    assert window.worker is worker
    assert not worker.cancel.is_set()
    assert window.composer.toPlainText() == ""
    assert window.store.pending_steering(window.session_id)[0]["text"].startswith(
        "Please use"
    )
    assert initial.content == "Original response."
    assert window.stream_message is None
    window.composer.setPlainText("Unsent follow-up draft")
    release.set()
    spin(app, lambda: window.worker is None)
    assert window.stream_message.content == "Updated response."
    assert initial.content == "Original response."
    assert "next prompt" in window.steering_status.text()
    assert window.composer.toPlainText() == "Unsent follow-up draft"
    assert len(engine.calls) == 1
    window.close()


def test_live_attachments_preserve_draft_and_do_not_queue(tmp_path):
    app, window, engine = setup_window(tmp_path)
    release = begin_held_chat(app, window, engine)
    file = tmp_path / "synthetic.txt"
    file.write_text("Synthetic fixture")
    window.add_attachments([str(file)])
    window.composer.setPlainText("See attached file")
    window.send()
    assert window.composer.toPlainText() == "See attached file"
    assert window.attachments == [str(file)]
    assert not window.store.pending_steering(window.session_id)
    assert "text only" in window.steering_status.text()
    release.set()
    spin(app, lambda: window.worker is None)
    window.close()


def test_media_work_and_other_selected_tasks_cannot_receive_updates(tmp_path):
    app, window, engine = setup_window(tmp_path)
    release = threading.Event()
    window.start_work(lambda cancel, emit: release.wait(2))
    assert not window.send_button.isEnabled()
    window.composer.setPlainText("Keep this draft")
    window.send()
    assert not window.store.updates
    assert window.composer.toPlainText() == "Keep this draft"
    release.set()
    spin(app, lambda: window.worker is None)
    release = begin_held_chat(app, window, engine)
    window.session_id = "another-task"
    window.update_send_mode()
    window.composer.setPlainText("Different task update")
    window.send()
    assert not window.store.updates
    assert not window.send_button.isEnabled()
    window.session_id = window._active_chat_session_id
    release.set()
    spin(app, lambda: window.worker is None)
    window.close()


def test_late_update_restarts_only_after_worker_release_with_frozen_options(tmp_path):
    app, window, engine = setup_window(tmp_path)
    release = begin_held_chat(app, window, engine, consume=False)
    first_run = engine.run

    def run(session_id, text, attachments, options, cancel, emit):
        if not engine.calls:
            return first_run(session_id, text, attachments, options, cancel, emit)
        engine.calls.append((session_id, text, attachments, options))
        window.store.consume(session_id)
        return {"status": "completed", "content": "Late update handled"}

    engine.run = run
    window.composer.setPlainText("Late instruction")
    window.send()
    window.allow_terminal.setChecked(True)
    window.composer.setPlainText("Preserved unsent draft")
    release.set()
    spin(app, lambda: len(engine.calls) == 2 and window.worker is None)
    assert engine.calls[1][1] == ""
    assert engine.calls[1][3]["allow_terminal"] is False
    assert engine.calls[1][0] == engine.calls[0][0]
    assert window.composer.toPlainText() == "Preserved unsent draft"
    window.close()


@pytest.mark.parametrize("outcome", ["stop", "failed", "delegated", "close"])
def test_pending_updates_survive_without_unwanted_auto_resume(tmp_path, outcome):
    app, window, engine = setup_window(tmp_path)
    release = begin_held_chat(
        app,
        window,
        engine,
        status="failed" if outcome == "failed" else "completed",
        consume=False,
    )
    window.composer.setPlainText("Retain this pending update")
    window.send()
    session_id = window.session_id
    if outcome == "stop":
        window.stop()
    elif outcome == "delegated":
        window._delegation_job_id = "synthetic-delegated"
    elif outcome == "close":
        window.close()
    release.set()
    spin(app, lambda: window.worker is None)
    QTest.qWait(20)
    app.processEvents()
    assert len(engine.calls) == 1
    assert window.store.pending_steering(session_id)
    assert "Continue" in window.steering_status.text()
    window.close()


def test_continue_preserves_unsent_draft_and_pending_updates_are_visible(tmp_path):
    app, window, engine = setup_window(tmp_path)
    session = window.store.create_session("Synthetic task", str(tmp_path))
    window.session_id = session["id"]
    window.store.queue_steering(session["id"], "Queued before restart")
    window.reload_sessions()
    window.select_task(window.task_list.item(0))
    assert "1 update(s) queued" in window.steering_status.text()
    window.composer.setPlainText("My unsent draft")

    def run(session_id, text, attachments, options, cancel, emit):
        engine.calls.append((session_id, text, attachments, options))
        window.store.consume(session_id)
        return {"status": "completed", "content": "Done"}

    engine.run = run
    window.resume()
    spin(app, lambda: window.worker is None)
    assert window.composer.toPlainText() == "My unsent draft"
    assert engine.calls[0][1] == "Continue"
    window.close()
