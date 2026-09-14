"""Iteration defaults and preference migration without GPU or external calls."""

import json

from test_ui import create_window


def test_old_mandatory_iteration_limit_becomes_unlimited(tmp_path, monkeypatch):
    """An old saved cap must not silently keep truncating resumed tasks."""
    monkeypatch.setenv("LOCAII_HOME", str(tmp_path))
    (tmp_path / "ui-settings.json").write_text(json.dumps({"max_iterations": 60}))
    _, window, _ = create_window(tmp_path)
    try:
        assert window.options()["max_iterations"] == 0
        assert window.max_iterations.text() == "Unlimited"
        window.max_iterations.setValue(7)
        window.save_preferences()
    finally:
        window.close()
    _, restored, _ = create_window(tmp_path)
    try:
        assert restored.options()["max_iterations"] == 7
    finally:
        restored.close()


def test_fresh_install_has_no_iteration_cutoff(tmp_path, monkeypatch):
    """New users start unlimited, and Stop remains the normal cancellation path."""
    monkeypatch.setenv("LOCAII_HOME", str(tmp_path))
    _, window, _ = create_window(tmp_path)
    try:
        assert window.options()["max_iterations"] == 0
    finally:
        window.close()


def test_loop_recovery_explanation_is_visible(tmp_path, monkeypatch):
    """Recovery must explain the changed next step instead of disappearing in logs."""
    monkeypatch.setenv("LOCAII_HOME", str(tmp_path))
    _, window, _ = create_window(tmp_path)
    try:
        explanation = (
            "Repeated read_text_file returned no new evidence. Reading the next page."
        )
        window.handle_event("recovery", {"text": explanation})
        assert explanation in window.activity.toPlainText()
        assert window.statusBar().currentMessage() == explanation
        assert not window.error_seen
    finally:
        window.close()


def test_missing_input_is_shown_as_attention_not_success(tmp_path, monkeypatch):
    """The deterministic input diagnosis must survive worker completion visibly."""
    monkeypatch.setenv("LOCAII_HOME", str(tmp_path))
    _, window, _ = create_window(tmp_path)
    try:
        explanation = "Current CSV has zero data rows. Provide an intact original."
        window.chat_finished({"status": "needs_attention", "text": explanation})
        assert window.error_seen
        assert not window._active_chat_succeeded
        assert explanation in window.activity.toPlainText()
        assert window.stream_message.content == explanation
    finally:
        window.close()
