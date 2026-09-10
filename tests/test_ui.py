"""Offscreen UI contracts: no GPU or external services are required."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from locaii.ui import MainWindow


class FakeStore:
    def __init__(self, root):
        self.root = root
        self.rows = []

    def sessions(self):
        return self.rows

    def create_session(self, title, workspace):
        item = {
            "id": "task-1",
            "title": title,
            "workspace": workspace,
            "status": "ready",
        }
        self.rows.append(item)
        return item

    def get_session(self, identity):
        return next(item for item in self.rows if item["id"] == identity)

    def messages(self, identity):
        return []

    def memory(self, identity):
        return "# Memory\n\nA verified checkpoint."


class FakeModels:
    def list_models(self):
        return [
            {
                "id": "qwen-test",
                "name": "Qwen test fixture",
                "path": "/tmp/fixture.gguf",
                "context": 32768,
                "gpu_layers": -1,
            }
        ]

    def register(self, *args, **kwargs):
        return {}

    def update(self, *args, **kwargs):
        return {}

    def stop(self):
        return None

    def start(self, identity, cancel, emit):
        emit("status", {"text": "Model fixture ready"})
        return {}


class FakeMedia:
    def list_workflows(self):
        return [{"name": "Synthetic image", "kind": "image"}]

    def jobs(self):
        return []


class FakeEngine:
    def __init__(self):
        self.calls = []

    def run(self, session_id, text, attachments, options, cancel, emit):
        self.calls.append((session_id, text, attachments, options))
        emit("status", {"text": "Verifying synthetic fixture"})
        emit("delta", {"text": "A verified "})
        emit("delta", {"text": "result."})
        emit("memory", {"text": "# Saved", "path": "/tmp/fixture.md"})
        return {"content": "A verified result.", "status": "completed"}


def create_window(tmp_path):
    app = QApplication.instance() or QApplication([])
    engine = FakeEngine()
    window = MainWindow(FakeStore(tmp_path), engine, FakeModels(), FakeMedia())
    window.show()
    app.processEvents()
    return app, window, engine


def wait_idle(app, window):
    deadline = time.monotonic() + 3
    while window.worker and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert window.worker is None
    app.processEvents()


def test_streaming_send_passes_options_and_attachments(tmp_path):
    app, window, engine = create_window(tmp_path)
    fixture = tmp_path / "input.txt"
    fixture.write_text("Synthetic fixture only")
    window.add_attachments([str(fixture), str(fixture)])
    assert len(window.attachments) == 1
    window.composer.setPlainText("Check this file")
    window.send()
    wait_idle(app, window)
    assert len(engine.calls) == 1
    assert engine.calls[0][2] == [str(fixture)]
    assert engine.calls[0][3]["allow_terminal"] is False
    assert engine.calls[0][3]["remote_enabled"] is False
    assert window.stream_message.content == "A verified result."
    assert "Ready" in window.run_status.text()
    assert not window.attachments
    window.close()


def test_remote_supervision_requires_opt_in(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.supervisor_provider.setCurrentIndex(
        window.supervisor_provider.findData("openai")
    )
    window.composer.setPlainText("A task")
    window.send()
    assert not engine.calls
    assert "external supervisor access" in window.error_label.text()
    assert window.pages.currentIndex() == 3
    window.close()


def test_errors_visible_and_preferences_are_private(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.show_error("Synthetic timeout")
    assert window.error_label.isVisible()
    assert "Needs attention" in window.run_status.text()
    window.save_preferences()
    assert (tmp_path / "ui-settings.json").is_file()
    assert window.raw_protection.isChecked()
    assert not window.raw_protection.isEnabled()
    window.close()


def test_pages_and_enter_submission(tmp_path):
    app, window, engine = create_window(tmp_path)
    assert window.pages.count() == 4
    for index in range(4):
        window.navigate(index)
        assert window.nav_buttons[index].isChecked()
    window.navigate(0)
    window.composer.setPlainText("Hello")
    QTest.keyClick(window.composer, Qt.Key_Return)
    wait_idle(app, window)
    assert len(engine.calls) == 1
    window.close()


def test_new_task_retains_workspace_but_clears_prior_messages(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.set_workspace(str(tmp_path))
    window.composer.setPlainText("Task one")
    window.send()
    wait_idle(app, window)
    window.new_task()
    assert window.session_id is None
    assert window.workspace == str(tmp_path)
    assert window.stream_message is None
    window.close()


def test_real_service_bootstrap_profile_save_and_review_export(tmp_path):
    """Exercise actual persistence, model registry, and supervision without inference."""
    from locaii.app import build_services

    store, engine, models, media = build_services(tmp_path)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(store, engine, models, media)
    window.show()
    app.processEvents()
    assert engine.media is media
    assert len(window._model_rows) >= 2
    window.models_list.setCurrentRow(0)
    window.model_name.setText("UI integration fixture")
    window.model_context.setValue(16384)
    window.model_vision.setChecked(False)
    window.mmproj.clear()
    window.save_model()
    assert models.list_models()[0]["name"] == "UI integration fixture"
    assert models.list_models()[0]["context"] == 16384
    session = store.create_session("Synthetic UI task", tmp_path / "workspace")
    store.append(session["id"], "user", "Inspect only synthetic files")
    window.session_id = session["id"]
    window.export_packet()
    wait_idle(app, window)
    assert (store.task_dir(session["id"]) / "supervisor-packet.json").is_file()
    assert "Review packet saved" in window.review_view.toPlainText()
    window.close()


def test_create_defaults_validate_with_real_workflow(tmp_path):
    """The UI random-seed sentinel must become a valid backend seed before dispatch."""
    from locaii.app import build_services

    store, engine, models, media = build_services(tmp_path)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(store, engine, models, media)
    dispatched = []

    def generate(kind, prompt, params, cancel, emit):
        workflow = next(
            item
            for item in media.list_workflows()
            if item["name"] == params["workflow"]
        )
        graph = media.prepare_graph(workflow, prompt, params, "ui-synthetic")
        dispatched.append((params, graph))
        return {"files": []}

    media.generate = generate
    window.media_prompt.setPlainText("A synthetic geometry test")
    window.generate()
    wait_idle(app, window)
    assert dispatched
    assert dispatched[0][0]["seed"] >= 0
    assert "no output files" in window.error_label.text()
    window.close()


def test_multimodal_history_never_renders_base64():
    content = [
        {"type": "text", "text": "Describe this attachment"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,SECRET_ENCODED_IMAGE"},
        },
    ]
    result = MainWindow.message_text(content)
    assert "Describe this attachment" in result
    assert "Attached image" in result
    assert "SECRET_ENCODED_IMAGE" not in result


def test_second_desktop_launch_activates_existing_instance(tmp_path):
    """Double-clicking twice must not create another Store or competing server."""
    import subprocess
    import sys

    command = [sys.executable, "-m", "locaii.app", "--home", str(tmp_path)]
    first = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "desktop.lock").exists() and time.monotonic() < deadline:
            if first.poll() is not None:
                raise AssertionError(first.communicate()[1].decode())
            time.sleep(0.02)
        second = subprocess.run(command, capture_output=True, timeout=5)
        assert second.returncode == 0, second.stderr.decode()
        assert first.poll() is None
    finally:
        first.terminate()
        first.communicate(timeout=5)


def test_pending_media_recovery_runs_off_ui_thread_and_shows_uncertainty(tmp_path):
    """Startup reconciliation must remain responsive and never claim unknown success."""
    import threading

    release = threading.Event()
    began = threading.Event()
    thread_ids = []

    class PendingMedia(FakeMedia):
        def jobs(self):
            return [
                {
                    "job_id": "synthetic",
                    "kind": "video",
                    "status": "running",
                    "files": [],
                }
            ]

        def recover_jobs(self):
            thread_ids.append(threading.get_ident())
            began.set()
            release.wait(2)
            return [
                {
                    "job_id": "synthetic",
                    "kind": "video",
                    "status": "unknown",
                    "files": [],
                    "recovery_error": "Synthetic engine connection timeout; outcome unconfirmed.",
                }
            ]

    app = QApplication.instance() or QApplication([])
    window = MainWindow(FakeStore(tmp_path), FakeEngine(), FakeModels(), PendingMedia())
    window.show()
    try:
        deadline = time.monotonic() + 2
        while not began.is_set() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(5)
        assert began.is_set()
        assert thread_ids[0] != threading.get_ident()
        assert window.worker is not None
        assert not window.generate_button.isEnabled()
        window.navigate(1)
        app.processEvents()
        assert window.pages.currentIndex() == 1
        release.set()
        wait_idle(app, window)
        assert "1 unknown" in window.media_status.text()
        assert "not automatically restarted" in window.media_status.text()
        assert window.media_results.item(0).data(Qt.UserRole) is None
        window.open_selected_media()
        assert "unconfirmed" in window.media_status.text()
    finally:
        release.set()
        wait_idle(app, window)
        window.close()


def test_cpu_preset_is_explicit_and_saves_real_profile(tmp_path):
    from locaii.app import build_services

    services = build_services(tmp_path)
    app = QApplication.instance() or QApplication([])
    window = MainWindow(*services)
    before = services[2].list_models()[0]
    assert before["gpu_layers"] != 0
    window.models_list.setCurrentRow(0)
    window.cpu_preset()
    after = services[2].list_models()[0]
    assert after["gpu_layers"] == 0
    assert after["device"] == "none"
    assert after["context"] == 8192
    assert after["vision"] is False
    assert window.worker is None
    assert "CPU-only preset saved" in window.model_status.text()
    window.close()
