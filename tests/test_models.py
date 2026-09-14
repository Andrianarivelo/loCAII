"""Backend contracts without loading production weights or using GPU memory."""

import json
import threading
from unittest.mock import Mock
import pytest
from locaii.models import ModelManager, ModelError, ModelCancelled


@pytest.fixture
def manager(tmp_path):
    return ModelManager(tmp_path / "models.json", tmp_path / "state")


def ready(manager):
    manager._process = Mock(pid=456)
    manager._process.poll.return_value = None
    manager._model = manager.list_models()[0]
    manager._phase = "ready"
    return manager


def stream(parts):
    response = Mock(ok=True)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_lines.return_value = [
        b"data: " + json.dumps(p).encode() for p in parts
    ] + [b"data: [DONE]"]
    return response


def test_registry_is_seeded_without_copying_weights(manager):
    profiles = manager.list_models()
    assert len(profiles) == 3
    assert {p["context"] for p in profiles} == {32768}
    assert all("abliterated" in p["id"] for p in profiles)


def test_register_validates_and_deduplicates_path(manager, tmp_path):
    model = tmp_path / "local.gguf"
    model.write_bytes(b"GGUF")
    first = manager.register(model, "First")
    second = manager.register(model, "Renamed", 8192, 0)
    assert first["id"] == second["id"]
    assert len(manager.list_models()) == 4
    assert manager.list_models()[-1]["name"] == "Renamed"
    with pytest.raises(ValueError):
        manager.register(model, "Too large", context=999999)


def test_stream_collects_fragmented_native_tool_call(manager):
    ready(manager)
    manager._session.post = Mock(
        return_value=stream(
            [
                {"choices": [{"delta": {"content": "Checking. "}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_a",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"note.md"}'},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 70, "completion_tokens": 15},
                },
            ]
        )
    )
    deltas = []
    result = manager.chat(
        [{"role": "user", "content": "Read note"}], on_delta=deltas.append
    )
    assert result["message"]["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path":"note.md"}',
    }
    assert deltas == ["Checking. "]
    assert result["usage"]["prompt_tokens"] == 70
    assert manager.status()["state"] == "ready"
    assert manager._session.post.call_args.kwargs["timeout"] == (10, 600)


def test_truncated_stream_is_visible_error(manager):
    ready(manager)
    manager._session.post = Mock(
        return_value=stream([{"choices": [{"delta": {"content": "Partial"}}]}])
    )
    with pytest.raises(ModelError, match="completion marker"):
        manager.chat([{"role": "user", "content": "Hi"}])


def test_http_error_is_visible(manager):
    ready(manager)
    response = stream([])
    response.ok, response.status_code, response.text = False, 400, "context overflow"
    manager._session.post = Mock(return_value=response)
    with pytest.raises(ModelError, match="context overflow"):
        manager.chat([])


def test_cancel_before_start_or_prediction_does_not_run(manager):
    event = threading.Event()
    event.set()
    with pytest.raises(ModelCancelled):
        manager.start(manager.list_models()[0]["id"], cancel=event)
    with pytest.raises(ModelCancelled):
        manager.chat([], cancel=event)


def test_count_includes_rendered_template_and_tools(manager):
    ready(manager)
    rendered, tokenized = Mock(), Mock()
    rendered.json.return_value = {"prompt": "<system> tool schemas <user>Hello"}
    tokenized.json.return_value = {"tokens": list(range(42))}
    manager._session.post = Mock(side_effect=[rendered, tokenized])
    tool = {"type": "function", "function": {"name": "read"}}
    assert manager.count_tokens([{"role": "user", "content": "Hello"}], [tool]) == 42
    assert manager._session.post.call_args_list[0].kwargs["json"]["tools"] == [tool]
    assert (
        manager._session.post.call_args_list[1]
        .kwargs["json"]["content"]
        .startswith("<system>")
    )


def test_stop_terminates_only_owned_process(manager):
    ready(manager)
    owned = manager._process
    manager.stop()
    owned.kill.assert_called_once()
    owned.wait.assert_called_once_with(timeout=5)
    assert not manager.status()["running"]


def test_no_inference_without_loaded_model(manager):
    with pytest.raises(ModelError, match="load a model"):
        manager.chat([])


def test_cancel_during_blocked_prediction_interrupts_owned_server(manager):
    """A stalled socket must not make the UI Stop control wait two minutes."""
    ready(manager)
    owned = manager._process
    event = threading.Event()
    terminated = threading.Event()
    owned.kill.side_effect = terminated.set
    response = stream([])

    def wait_then_disconnect(*args, **kwargs):
        event.set()
        assert terminated.wait(2), "Cancellation did not interrupt the backend"
        import requests

        raise requests.ConnectionError("Server interrupted")

    manager._session.post = wait_then_disconnect
    with pytest.raises(ModelCancelled):
        manager.chat([], cancel=event)
    owned.kill.assert_called_once()
    owned.wait.assert_called_once_with(timeout=5)
    assert manager._process is None
    assert manager.status()["state"] == "stopped"


def test_busy_port_does_not_spawn_or_kill_foreign_service(
    manager, monkeypatch, tmp_path
):
    import socket

    model = tmp_path / "test.gguf"
    model.write_bytes(b"GGUF")
    profile = manager.register(model, "Fixture", gpu_layers=0)
    monkeypatch.setattr(manager, "_command", lambda _: ["/bin/false"])
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        manager.port = listener.getsockname()[1]
        with pytest.raises(ModelError, match="already occupied"):
            manager.start(profile["id"])
        assert manager._process is None
        assert listener.fileno() >= 0


def test_register_does_not_modify_weight_bytes(manager, tmp_path):
    model = tmp_path / "fixture.gguf"
    original = b"GGUF\x00fixture-weights"
    model.write_bytes(original)
    manager.register(model, "Fixture")
    assert model.read_bytes() == original


def test_profile_update_validates_and_preserves_identity(manager):
    profile = manager.list_models()[0]
    changed = manager.update(profile["id"], context=16384, gpu_layers=20, vision=False)
    assert changed["id"] == profile["id"]
    assert changed["context"] == 16384
    assert manager.list_models()[0]["gpu_layers"] == 20
    with pytest.raises(ValueError):
        manager.update(profile["id"], context=123)
    with pytest.raises(ValueError):
        manager.update(profile["id"], command="arbitrary")


def test_image_count_includes_reserved_embedding_budget(manager):
    ready(manager)
    rendered, tokenized = Mock(), Mock()
    rendered.json.return_value = {"prompt": "image-placeholder"}
    tokenized.json.return_value = {"tokens": [1, 2, 3]}
    manager._session.post = Mock(side_effect=[rendered, tokenized])
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,fixture"},
                }
            ],
        }
    ]
    assert manager.count_tokens(messages) == 2051


def test_gpu_contention_is_actionable_before_allocating(manager, monkeypatch, tmp_path):
    import subprocess

    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"GGUF")
    profile = {"path": str(weight), "gpu_layers": -1, "context": 32768}
    manager._device = "Vulkan1"
    monkeypatch.setattr("locaii.models.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "locaii.models.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "0, NVIDIA Fixture, 512, 8192\n", ""
        ),
    )
    with pytest.raises(ModelError, match="Wait for the other GPU task"):
        manager._check_gpu_memory(profile)
    assert manager._process is None


def test_cpu_profile_never_needs_nvidia_tools(manager, monkeypatch):
    monkeypatch.setattr(
        "locaii.models.shutil.which",
        lambda _: (_ for _ in ()).throw(AssertionError("NVIDIA query not needed")),
    )
    manager._check_gpu_memory({"gpu_layers": 0, "device": "none"})


def test_missing_nvidia_utility_does_not_disable_vulkan(manager, monkeypatch):
    monkeypatch.setattr("locaii.models.shutil.which", lambda _: None)
    manager._check_gpu_memory({"gpu_layers": -1})


def test_available_nvidia_headroom_allows_load(manager, monkeypatch, tmp_path):
    import subprocess

    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"GGUF")
    manager._device = "Vulkan1"
    monkeypatch.setattr("locaii.models.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "locaii.models.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "0, NVIDIA Fixture, 7800, 8192\n", ""
        ),
    )
    manager._check_gpu_memory({"path": str(weight), "gpu_layers": -1, "context": 32768})


def test_explicit_integrated_gpu_does_not_query_nvidia_headroom(manager, monkeypatch):
    manager._device = "Vulkan1"
    monkeypatch.setattr(
        "locaii.models.shutil.which",
        lambda _: (_ for _ in ()).throw(AssertionError("NVIDIA query not needed")),
    )
    manager._check_gpu_memory({"gpu_layers": -1, "device": "Vulkan0"})


def test_start_never_reuses_a_draining_server(manager, monkeypatch, tmp_path):
    path = tmp_path / "fixture.gguf"
    path.write_bytes(b"GGUF")
    registered = manager.register(path, "Fixture", gpu_layers=0)
    profile = next(p for p in manager.list_models() if p["id"] == registered["id"])
    ready(manager)
    manager._model = profile
    manager._phase = "stopping"
    monkeypatch.setattr(manager, "_command", lambda _: ["/unused"])
    cleanup = Mock(side_effect=ModelError("Draining process cleanup attempted"))
    monkeypatch.setattr(manager, "stop", cleanup)
    with pytest.raises(ModelError, match="Draining process cleanup"):
        manager.start(profile["id"])
    cleanup.assert_called_once()


def test_zero_gpu_layers_explicitly_disables_device_offload(manager, monkeypatch):
    from pathlib import Path

    manager._device = "Vulkan1"
    monkeypatch.setattr(manager, "_runtime", lambda: Path("/unused"))
    profile = manager.list_models()[0]
    profile["gpu_layers"] = 0
    command = manager._command(profile)
    assert command[command.index("--device") + 1] == "none"


def test_interrupt_unblocks_token_count_without_waiting_for_operation_lock(manager):
    """Stop interrupts prefill/template HTTP even outside chat's cancel watcher."""
    import requests

    ready(manager)
    owned = manager._process
    entered, killed = threading.Event(), threading.Event()
    owned.kill.side_effect = killed.set
    errors = []

    def blocked_request(*args, **kwargs):
        entered.set()
        assert killed.wait(2), "Stop failed to signal the owned backend"
        raise requests.ConnectionError("Owned server stopped")

    def count():
        try:
            manager.count_tokens([])
        except Exception as error:
            errors.append(error)

    manager._session.post = blocked_request
    worker = threading.Thread(target=count)
    worker.start()
    assert entered.wait(2)
    manager.interrupt()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ModelCancelled)
    manager.stop()
    assert manager.status()["state"] == "stopped"
    assert manager._model is None


def test_stop_during_device_discovery_cannot_spawn_late_server(
    manager, monkeypatch, tmp_path
):
    """A cancelled startup cannot allocate VRAM after Stop returned to the UI."""
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"GGUF")
    profile = manager.register(model, "Fixture", gpu_layers=0)
    entered, release = threading.Event(), threading.Event()
    errors = []

    def command(_):
        entered.set()
        assert release.wait(2)
        return ["/unused"]

    def start():
        try:
            manager.start(profile["id"])
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(manager, "_command", command)
    spawn = Mock(side_effect=AssertionError("Cancelled startup spawned a server"))
    monkeypatch.setattr("locaii.models.subprocess.Popen", spawn)
    worker = threading.Thread(target=start)
    worker.start()
    assert entered.wait(2)
    manager.interrupt()
    release.set()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ModelCancelled)
    spawn.assert_not_called()
    assert manager.status()["state"] == "stopped"
    assert manager._log is None


@pytest.mark.parametrize("phase", ["ready", "error", "stopped"])
def test_stop_reaps_real_owned_idle_child_and_clears_error(manager, phase):
    """Only a fresh test-owned child is killed; no model weights or GPU are used."""
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    manager._process = child
    manager._model = manager.list_models()[0]
    manager._phase, manager._error = phase, "previous error"
    try:
        manager.stop()
        assert child.poll() is not None
        assert manager.status()["state"] == "stopped"
        assert manager.status()["error"] == ""
        assert manager._process is None
        manager.stop()  # Repeated clicks after release remain harmless.
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)


def test_interrupt_during_health_cannot_publish_ready(manager, monkeypatch, tmp_path):
    """A late successful health response must not undo the stopped state."""
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"GGUF")
    profile = manager.register(model, "Fixture", gpu_layers=0)
    child = Mock(pid=654)
    child.poll.return_value = None
    monkeypatch.setattr(manager, "_command", lambda _: ["/unused"])
    monkeypatch.setattr("locaii.models.subprocess.Popen", Mock(return_value=child))

    def health(*args, **kwargs):
        manager.interrupt()
        return Mock(status_code=200)

    manager._session.get = health
    with pytest.raises(ModelCancelled):
        manager.start(profile["id"])
    assert manager.status()["state"] == "stopped"
    assert manager._process is None
    child.wait.assert_called()


def test_direct_interrupt_of_chat_reaps_backend_without_cancel_event(manager):
    """The explicit unload API works even when no prediction event was supplied."""
    import requests

    ready(manager)
    owned = manager._process

    def interrupted_post(*args, **kwargs):
        manager.interrupt()
        raise requests.ConnectionError("Stopped")

    manager._session.post = interrupted_post
    with pytest.raises(ModelCancelled):
        manager.chat([])
    owned.wait.assert_called_once_with(timeout=5)
    assert manager._process is None
    assert manager.status()["state"] == "stopped"
