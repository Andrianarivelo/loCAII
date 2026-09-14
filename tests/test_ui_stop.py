"""Stop contracts using held synthetic workers, never real models or processes."""

import threading
import time

from PySide6.QtTest import QTest

from test_ui import create_window


def spin(app, condition, timeout=3):
    """Pump queued signals until the deterministic worker reaches its checkpoint."""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert condition()
    app.processEvents()


def test_idle_stop_unloads_asynchronously_and_blocks_new_work(tmp_path):
    """An idle/error task still owns a model; Stop must unload it without freezing."""
    app, window, engine = create_window(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []
    ui_thread = threading.get_ident()

    def unload():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(3)

    window.models.stop = unload
    window.models.interrupt = lambda: calls.append("interrupt")
    try:
        window.show_error("Iteration limit reached")
        assert window.stop_button.isEnabled()
        window.stop()
        spin(app, entered.is_set)
        assert calls[0] == "interrupt"
        assert calls[1] != ui_thread
        assert not window.send_button.isEnabled()
        assert not window.load_model_button.isEnabled()
        window.composer.setPlainText("Do not start while unloading")
        window.send()
        assert not engine.calls
        window.stop()  # Multiple clicks cannot race multiple unload workers.
        assert len(calls) == 2
        release.set()
        spin(app, lambda: not window._stop_requested)
        assert window.send_button.isEnabled()
        assert window.stop_button.isEnabled()
        assert "Stopped" in window.run_status.text()
        assert not window.error_label.isVisible()
    finally:
        release.set()
        spin(app, lambda: window._stop_worker is None)
        window.close()


def test_stop_waits_for_task_and_cleanup_before_close(tmp_path):
    """Finishing the task first cannot destroy the still-running cleanup QThread."""
    app, window, _ = create_window(tmp_path)
    task_entered, cleanup_entered = threading.Event(), threading.Event()
    release_cleanup = threading.Event()
    interrupt_saw_cancel = []

    def action(cancel, emit):
        task_entered.set()
        assert cancel.wait(3)
        return {"status": "interrupted"}

    def cleanup():
        cleanup_entered.set()
        assert release_cleanup.wait(3)

    window.models.interrupt = lambda: interrupt_saw_cancel.append(
        window.worker.cancel.is_set()
    )
    window.models.stop = cleanup
    try:
        window.start_work(action)
        spin(app, task_entered.is_set)
        window.stop()
        spin(app, lambda: cleanup_entered.is_set() and window.worker is None)
        assert interrupt_saw_cancel == [True]
        assert window._stop_requested
        assert not window.send_button.isEnabled()
        window.close()
        assert window.isVisible()
        release_cleanup.set()
        spin(app, lambda: not window.isVisible())
        assert window._stop_worker is None
    finally:
        release_cleanup.set()
        spin(app, lambda: window.worker is None and window._stop_worker is None)
        window.close()


def test_unload_error_does_not_claim_success(tmp_path):
    """A failed model unload leaves an actionable error and permits retry."""
    app, window, _ = create_window(tmp_path)

    def fail():
        raise RuntimeError("Synthetic cleanup failure")

    window.models.stop = fail
    try:
        window.stop()
        spin(app, lambda: not window._stop_requested)
        assert "Synthetic cleanup failure" in window.error_label.text()
        assert "Needs attention" in window.run_status.text()
        assert window.stop_button.isEnabled()
    finally:
        window.close()


def test_cleanup_finishing_first_does_not_allow_a_new_task(tmp_path):
    """Keep serialization until the cancelled task has finished its own checkpoint."""
    app, window, _ = create_window(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def action(cancel, emit):
        entered.set()
        assert release.wait(3)
        return {"status": "interrupted"}

    try:
        window.start_work(action)
        spin(app, entered.is_set)
        window.stop()
        spin(app, lambda: window._stop_worker is None)
        assert window.worker is not None
        assert window.worker.cancel.is_set()
        assert window._stop_requested
        assert not window.send_button.isEnabled()
        release.set()
        spin(app, lambda: not window._stop_requested)
        assert window.worker is None
        assert window.send_button.isEnabled()
    finally:
        release.set()
        spin(app, lambda: window.worker is None and window._stop_worker is None)
        window.close()
