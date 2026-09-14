"""Supervisor controls exercise real Qt workers using local, synthetic service doubles."""

import sys
import types

import pytest

from PySide6.QtWidgets import QApplication

from test_ui import create_window, wait_idle


class FakeHub:
    """Model a connection and its queue without exposing ports or loading a model."""

    def __init__(self, home):
        self.home = home
        self.jobs = []
        self.events = []
        self.results = []
        self.failures = []
        self.connection = None
        self.disconnected = False
        self.cancel = False

    def connect(self, workspace, options, model):
        self.connection = (workspace, options, model)
        return self.home / "connection.json"

    def status(self):
        return {"connected": True, "queued": len(self.jobs)}

    def take_next(self):
        return self.jobs.pop(0) if self.jobs else None

    def bind_session(self, identity, session_id):
        self.bound = (identity, session_id)

    def record_event(self, identity, event, payload):
        self.events.append((identity, event, payload))

    def complete(self, identity, result):
        self.results.append((identity, result))

    def fail(self, identity, error):
        self.failures.append((identity, error))

    def cancel_requested(self, identity):
        return self.cancel

    def disconnect(self):
        self.disconnected = True
        self.jobs.clear()


class FakeLauncher:
    """Capture the launch contract without starting or charging an external CLI."""

    calls = []

    def __init__(self, home):
        self.home = home

    def prepare(self, *args):
        self.calls.append(("prepare", args))
        return {"instructions": "Run the isolated supervisor command in your terminal."}

    def open(self, *args):
        self.calls.append(("open", args))
        return {"status": "opened", "instructions": "Synthetic connection instructions"}


def install_services(monkeypatch):
    """Lazy imports ensure ordinary local chat never needs the connection service."""
    monkeypatch.setitem(
        sys.modules, "locaii.delegation", types.SimpleNamespace(DelegationHub=FakeHub)
    )
    monkeypatch.setitem(
        sys.modules,
        "locaii.supervisor_launch",
        types.SimpleNamespace(SupervisorLauncher=FakeLauncher),
    )
    FakeLauncher.calls = []


def test_open_supervisor_only_authorizes_frozen_local_connection(tmp_path, monkeypatch):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.set_workspace(str(tmp_path))
    window.read_all_files.setChecked(False)
    window.open_supervisor()
    hub = window._delegation_hub
    assert hub.connection[0] == str(tmp_path)
    assert hub.connection[1]["read_all_files"] is False
    assert hub.connection[1]["allow_terminal"] is False
    assert hub.connection[1]["supervisor_provider"] == "off"
    assert hub.connection[1]["remote_enabled"] is False
    assert not window.remote_enabled.isChecked()
    assert window.supervisor_provider.currentData() == "off"
    assert FakeLauncher.calls[0][0] == "open"
    assert "Waiting" in window.supervisor_connection_status.text()
    window.read_all_files.setChecked(True)
    assert hub.connection[1]["read_all_files"] is False
    window.close()
    assert hub.disconnected


def test_existing_connection_instructions_do_not_launch_a_cli(tmp_path, monkeypatch):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.copy_supervisor_connection()
    assert FakeLauncher.calls[0][0] == "prepare"
    assert "isolated supervisor" in QApplication.clipboard().text()
    window.close()


def test_advanced_review_settings_are_collapsed_and_remain_available(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.navigate(3)
    assert not window.advanced_supervision.isVisible()
    window.advanced_supervision_toggle.click()
    assert window.advanced_supervision.isVisible()
    window.supervisor_provider.setCurrentIndex(
        window.supervisor_provider.findData("codex-cli")
    )
    assert window.token_budget.value() == 200000
    window.token_budget.setValue(40000)
    window.supervisor_provider.setCurrentIndex(
        window.supervisor_provider.findData("claude-cli")
    )
    assert window.token_budget.value() == 40000
    window.close()


def test_delegation_preserves_draft_and_uses_frozen_permissions(tmp_path, monkeypatch):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.composer.setPlainText("My unsent personal draft")
    attachment = tmp_path / "draft.txt"
    attachment.write_text("synthetic")
    window.add_attachments([str(attachment)])
    window.open_supervisor()
    window.delegation_timer.stop()
    hub = window._delegation_hub
    hub.jobs.append(
        {
            "id": "job-1",
            "task": "Check synthetic fixture",
            "title": "Check fixture",
            "workspace": str(tmp_path),
            "options": {"model_id": "qwen-test", "allow_terminal": False},
        }
    )
    window.poll_delegation()
    wait_idle(app, window)
    assert engine.calls[0][1] == "Check synthetic fixture"
    assert engine.calls[0][2] == []
    assert engine.calls[0][3]["supervisor_provider"] == "off"
    assert engine.calls[0][3]["remote_enabled"] is False
    assert window.composer.toPlainText() == "My unsent personal draft"
    assert window.attachments == [str(attachment)]
    assert window.session_id is None
    assert hub.results[0][1]["status"] == "completed"
    assert hub.events
    assert window._delegation_job_id is None
    assert window.task_list.count() == 1
    window.close()


def test_queue_waits_for_user_worker_and_disconnect_keeps_it_running(
    tmp_path, monkeypatch
):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.open_supervisor()
    window.delegation_timer.stop()
    hub = window._delegation_hub
    hub.jobs.append({"id": "queued"})
    sentinel = types.SimpleNamespace(cancel=types.SimpleNamespace(set=lambda: None))
    window.worker = sentinel
    window.poll_delegation()
    assert hub.jobs == [{"id": "queued"}]
    assert not engine.calls
    window.disconnect_supervisor()
    assert window.worker is sentinel
    assert hub.disconnected
    assert not hub.jobs
    window.worker = None
    window.close()


def test_delegated_failure_is_reported_to_supervisor(tmp_path, monkeypatch):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.open_supervisor()
    window.delegation_timer.stop()
    hub = window._delegation_hub
    hub.jobs.append(
        {
            "id": "job-failure",
            "task": "Synthetic failure",
            "workspace": str(tmp_path),
            "options": {},
        }
    )

    def fail(*args):
        raise RuntimeError("Synthetic worker failure")

    engine.run = fail
    window.poll_delegation()
    wait_idle(app, window)
    assert hub.failures == [("job-failure", "Synthetic worker failure")]
    assert not hub.results
    assert window._delegation_job_id is None
    window.close()


def test_connection_is_reused_and_only_claimed_client_is_shown_connected(
    tmp_path, monkeypatch
):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.set_workspace(str(tmp_path))
    window.open_supervisor()
    window.delegation_timer.stop()
    hub = window._delegation_hub
    first_connection = hub.connection
    window.set_workspace(str(tmp_path.parent))
    window.copy_supervisor_connection()
    window.delegation_timer.stop()
    assert hub.connection is first_connection
    assert FakeLauncher.calls[-1][1][2] == str(tmp_path)
    window.poll_delegation()
    assert "Waiting" in window.supervisor_connection_status.text()
    hub.status = lambda: {"connected": True, "last_client_seen": 1234, "queued": 0}
    window.poll_delegation()
    assert window.supervisor_connection_status.text().startswith("Supervisor contacted")
    window.close()


def test_remote_cancel_only_cancels_active_delegated_worker(tmp_path, monkeypatch):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.open_supervisor()
    window.delegation_timer.stop()
    hub = window._delegation_hub
    hub.cancel = True
    hub.jobs.append(
        {
            "id": "job-cancel",
            "task": "Wait for cancellation",
            "workspace": str(tmp_path),
            "options": {},
        }
    )

    def until_cancelled(session_id, task, attachments, options, cancel, emit):
        if not cancel.wait(2):
            raise RuntimeError("Cancellation was not delivered")
        return {"status": "interrupted", "content": "Checkpoint retained"}

    engine.run = until_cancelled
    window.poll_delegation()
    window.poll_delegation()
    wait_idle(app, window)
    assert hub.results[0][1]["status"] == "interrupted"
    window.close()


@pytest.mark.parametrize("action", ["open", "prepare"])
def test_launch_errors_preserve_an_existing_connection_and_jobs(
    tmp_path, monkeypatch, action
):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.open_supervisor()
    window.delegation_timer.stop()
    hub = window._delegation_hub
    scope = window._supervisor_scope
    hub.jobs.append({"id": "must-remain-queued"})

    def fail(*args):
        raise RuntimeError("Synthetic missing terminal")

    monkeypatch.setattr(FakeLauncher, action, fail)
    if action == "open":
        window.open_supervisor()
    else:
        window.copy_supervisor_connection()
    window.delegation_timer.stop()
    assert not hub.disconnected
    assert window._supervisor_scope is scope
    assert hub.jobs == [{"id": "must-remain-queued"}]
    assert "Synthetic missing terminal" in window.supervisor_launch_error.text()
    assert not window.supervisor_launch_error.isHidden()
    window.close()


def test_first_launch_failure_revokes_only_the_new_connection(tmp_path, monkeypatch):
    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)

    def fail(*args):
        raise RuntimeError("Synthetic missing CLI")

    monkeypatch.setattr(FakeLauncher, "open", fail)
    window.open_supervisor()
    assert window._delegation_hub.disconnected
    assert window._supervisor_scope is None
    assert "Synthetic missing CLI" in window.supervisor_launch_error.text()
    assert not window.delegation_timer.isActive()
    window.close()


def test_saved_and_explicit_small_allowances_are_preserved(tmp_path):
    import json

    (tmp_path / "ui-settings.json").write_text(
        json.dumps(
            {"supervisor_token_budget": 12000, "supervisor_provider": "codex-cli"}
        )
    )
    app, window, engine = create_window(tmp_path)
    assert window.token_budget.value() == 12000
    assert not window.supervisor_allowance_hint.isHidden()
    window.supervisor_provider.setCurrentIndex(
        window.supervisor_provider.findData("off")
    )
    window.token_budget.setValue(9000)
    window.supervisor_provider.setCurrentIndex(
        window.supervisor_provider.findData("claude-cli")
    )
    assert window.token_budget.value() == 9000
    assert not window.supervisor_allowance_hint.isHidden()
    window.token_budget.setValue(60000)
    assert window.supervisor_allowance_hint.isHidden()
    window.close()


@pytest.mark.parametrize("action", ["open", "copy", "save"])
def test_primary_supervisor_choice_roundtrips_without_remote_enable(
    tmp_path, monkeypatch, action
):
    import json

    install_services(monkeypatch)
    app, window, engine = create_window(tmp_path)
    window.supervisor_cli.setCurrentIndex(window.supervisor_cli.findData("claude-cli"))
    if action == "open":
        window.open_supervisor()
    elif action == "copy":
        window.copy_supervisor_connection()
    else:
        window.save_preferences()
    assert not window.remote_enabled.isChecked()
    assert window.supervisor_provider.currentData() == "off"
    saved = json.loads((tmp_path / "ui-settings.json").read_text())
    assert saved["supervisor_cli"] == "claude-cli"
    assert saved.get("remote_enabled", False) is False
    window.close()
    app, restored, engine = create_window(tmp_path)
    assert restored.supervisor_cli.currentData() == "claude-cli"
    assert not restored.remote_enabled.isChecked()
    assert restored.supervisor_provider.currentData() == "off"
    restored.close()


def test_review_scope_defaults_to_tool_work_and_roundtrips(tmp_path):
    app, window, engine = create_window(tmp_path)
    assert window.options()["supervisor_scope"] == "tasks"
    window.supervisor_scope.setCurrentIndex(window.supervisor_scope.findData("all"))
    window.save_preferences()
    window.close()
    app, restored, engine = create_window(tmp_path)
    assert restored.options()["supervisor_scope"] == "all"
    restored.close()


def test_review_skipped_retains_answer_without_error_or_review_claim(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.handle_event("delta", {"text": "A useful local answer."})
    answer = window.stream_message
    review_before = window.review_view.toPlainText()
    status_before = window.run_status.text()
    window.handle_event(
        "review_skipped",
        {
            "reason": "allowance",
            "text": "Review allowance reached. Local work continues.",
            "reviewed": False,
        },
    )
    assert window.stream_message is answer
    assert answer.content == "A useful local answer."
    assert not window.error_seen
    assert window.error_label.isHidden()
    assert window.review_view.toPlainText() == review_before
    assert window.run_status.text() == status_before
    assert "Review skipped" in window.activity.toPlainText()
    assert "Local work continues" in window.statusBar().currentMessage()
    window.handle_event("delta", {"text": " Still working."})
    assert answer.content == "A useful local answer. Still working."
    window.close()
