"""Offline regression tests for workflow controls, receipts and cancellation."""

import json
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from locaii.media import MediaCancelled, MediaService


@pytest.fixture
def service(tmp_path):
    """Construct a real adapter with isolated durable state and fake GPU manager."""
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"media": {"output_dir": str(tmp_path / "outputs"), "poll_seconds": 0}}
        )
    )
    result = MediaService(config, tmp_path / "state", Mock())
    result._check_gpu_available = Mock()
    return result


def test_builtin_controls(service):
    """Size, seed and frame controls reach the intended nodes only."""
    image, video = service.list_workflows()
    graph = service.prepare_graph(
        image, "A teapot", {"width": 512, "steps": 5, "seed": 19}, "test"
    )
    assert graph["4"]["inputs"]["text"] == "A teapot"
    assert graph["9"]["inputs"]["width"] == 512
    assert graph["10"]["inputs"]["seed"] == 19
    assert graph["13"]["inputs"]["filename_prefix"] == "loCAII/image/test"
    graph = service.prepare_graph(video, "Moving boat", {"frames": 9}, "test")
    assert graph["5"]["inputs"]["length"] == 9
    assert video["supports_image_input"]


@pytest.mark.parametrize(
    "params", [{"frames": 12}, {"width": 511}, {"steps": 0}, {"seed": -1}]
)
def test_invalid_controls(service, params):
    with pytest.raises(ValueError):
        service.prepare_graph(service.list_workflows()[1], "A boat", params, "test")


def test_workflow_registration(service, tmp_path):
    """Imported API graphs remain modular and discover positive prompt edges."""
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad"}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "seed": 4}},
    }
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps(graph))
    entry = service.register_workflow("Custom image", source, "image")
    assert entry["mapping"]["prompt"] == ["1", "text"]
    assert Path(entry["path"]).is_file()
    with pytest.raises(ValueError, match="already exists"):
        service.register_workflow("Custom image", source, "image")


def test_reject_visual_editor_workflow(service, tmp_path):
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps({"nodes": [], "links": []}))
    with pytest.raises(ValueError, match="API format"):
        service.register_workflow("Invalid", source, "image")


def test_output_path_traversal(service):
    with pytest.raises(RuntimeError, match="outside"):
        service._files(
            {
                "outputs": {
                    "1": {"images": [{"type": "output", "filename": "../../secret"}]}
                }
            }
        )


def test_cancel_scoped(service):
    service._api = Mock(return_value={})
    service._cancel_prompt("own-prompt")
    service._api.assert_called_once_with("/api/jobs/own-prompt/cancel", {})


def test_cancel_old_engine_never_global_interrupt(service):
    service._api = Mock(side_effect=[RuntimeError("404"), {}])
    with pytest.raises(RuntimeError, match="may still be running"):
        service._cancel_prompt("own-prompt")
    assert service._api.call_args_list[1].args == ("/queue", {"delete": ["own-prompt"]})


def test_busy_engine_not_submitted(service):
    service._ensure_service = Mock()
    service._api = Mock(return_value={"queue_running": [[1, "someone-else"]]})
    with pytest.raises(RuntimeError, match="busy"):
        service.generate("image", "A teapot", {}, threading.Event(), Mock())
    assert all(call.args[0] != "/prompt" for call in service._api.call_args_list)
    assert service.jobs()[0]["status"] == "failed"


def test_success_receipt_and_real_output(service):
    """Success requires a nonempty engine-produced file and durable full history."""
    root = Path(service.config["output_dir"])
    root.mkdir()
    (root / "result.png").write_bytes(b"test fixture")
    result = {
        "status": {"status_str": "success"},
        "outputs": {"1": {"images": [{"type": "output", "filename": "result.png"}]}},
    }

    def api(route, payload=None):
        if route == "/prompt":
            return {"prompt_id": "own-prompt"}
        if route.startswith("/history/"):
            return {"own-prompt": result}
        return {}

    service._api = Mock(side_effect=api)
    service._ensure_service = Mock()
    events = Mock()
    report = service.generate("image", "A teapot", {}, threading.Event(), events)
    assert report["status"] == "completed"
    assert report["files"] == [str(root / "result.png")]
    assert len(service.jobs()) == 1
    assert list((service.state_dir / "jobs").glob("*.history.json"))
    service.model_manager.stop.assert_called_once()


def test_cancel_before_gpu_release(service):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(MediaCancelled):
        service.generate("image", "A teapot", {}, cancel, Mock())
    service.model_manager.stop.assert_not_called()
    assert service.jobs()[0]["status"] == "cancelled"


def test_external_work_not_killed_during_cleanup(service):
    process = Mock()
    service._process = process
    service._api = Mock(return_value={"queue_pending": [[1, "foreign-prompt"]]})
    service._release("own-prompt")
    process.terminate.assert_not_called()


def test_uploaded_wan_image_graph(service, tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64)).save(source)
    service.config["comfy_dir"] = str(tmp_path / "comfy")
    graph = service.prepare_graph(
        service.list_workflows()[1], "Animate", {"image_path": str(source)}, "job"
    )
    assert graph["5"]["inputs"]["start_image"] == ["locaii_source_image", 0]
    assert (tmp_path / "comfy/input/locaii/job.png").exists()
    with pytest.raises(ValueError, match="does not support"):
        service.prepare_graph(
            service.list_workflows()[0], "Animate", {"image_path": str(source)}, "job"
        )


def test_history_error_cancels_only_submitted_job(service):
    """A broken monitoring request must not silently leave an active job behind."""

    def api(route, payload=None):
        if route == "/prompt":
            return {"prompt_id": "own-prompt"}
        if route.startswith("/history/"):
            raise RuntimeError("Monitoring connection failed")
        return {}

    service._api = Mock(side_effect=api)
    service._ensure_service = Mock()
    with pytest.raises(RuntimeError, match="Monitoring"):
        service.generate("image", "A teapot", {}, threading.Event(), Mock())
    assert any(
        call.args == ("/api/jobs/own-prompt/cancel", {})
        for call in service._api.call_args_list
    )
    assert not any(call.args[0] == "/interrupt" for call in service._api.call_args_list)
    assert service.jobs()[0]["status"] == "failed"


def test_success_without_output_is_not_reported_as_completed(service):
    def api(route, payload=None):
        if route == "/prompt":
            return {"prompt_id": "own-prompt"}
        if route.startswith("/history/"):
            return {"own-prompt": {"status": {"status_str": "success"}, "outputs": {}}}
        return {}

    service._api = Mock(side_effect=api)
    service._ensure_service = Mock()
    with pytest.raises(RuntimeError, match="no nonempty output"):
        service.generate("image", "A teapot", {}, threading.Event(), Mock())
    assert service.jobs()[0]["status"] == "failed"


def _pending_receipt(service, job_id="old-job", prompt_id="old-prompt"):
    """Persist an unfinished job exactly as a crashed app would leave it."""
    from locaii.media import _atomic_json

    record = {
        "job_id": job_id,
        "prompt_id": prompt_id,
        "status": "running",
        "started_at": 1,
        "files": [],
    }
    _atomic_json(service.state_dir / "jobs" / (job_id + ".json"), record)
    return record


def test_recover_completed_during_gui_absence(service):
    root = Path(service.config["output_dir"])
    root.mkdir()
    (root / "result.png").write_bytes(b"fixture")
    _pending_receipt(service)
    history = {
        "old-prompt": {
            "status": {"status_str": "success"},
            "outputs": {
                "1": {"images": [{"type": "output", "filename": "result.png"}]}
            },
        }
    }
    service._api = Mock(side_effect=[{}, history, {}])
    result = service.recover_jobs()
    assert result[0]["status"] == "completed"
    assert result[0]["files"] == [str(root / "result.png")]
    assert service._api.call_count == 3
    assert service._api.call_args_list[0].kwargs == {"timeout": 2}


def test_recover_failure_and_running_foreign_work(service):
    _pending_receipt(service)
    service._api = Mock(
        side_effect=[
            {"queue_running": [[1, "foreign"]]},
            {"old-prompt": {"status": {"status_str": "error"}}},
        ]
    )
    assert service.recover_jobs()[0]["status"] == "failed"
    assert service._api.call_count == 2
    assert all(
        call.args[0] not in {"/free", "/interrupt"}
        for call in service._api.call_args_list
    )


def test_recover_missing_engine_is_interrupted(service):
    from urllib.error import URLError

    _pending_receipt(service)
    service._api = Mock(side_effect=URLError(ConnectionRefusedError("refused")))
    result = service.recover_jobs()[0]
    assert result["status"] == "interrupted"
    assert "unconfirmed" in result["recovery_error"]


def test_recover_timeout_preserves_unknown_outcome(service):
    _pending_receipt(service)
    service._api = Mock(side_effect=TimeoutError("temporary network timeout"))
    result = service.recover_jobs()[0]
    assert result["status"] == "unknown"
    assert result["files"] == []


def test_recover_batch_does_not_request_once_per_job(service):
    for index in range(100):
        _pending_receipt(service, f"job-{index}", f"prompt-{index}")
    service._api = Mock(side_effect=[{"queue_running": [[1, "prompt-1"]]}, {}])
    records = service.recover_jobs()
    assert service._api.call_count == 2
    assert (
        next(row for row in records if row["job_id"] == "job-1")["status"] == "running"
    )
    assert (
        next(row for row in records if row["job_id"] == "job-2")["status"]
        == "interrupted"
    )


def test_jobs_is_disk_only_even_for_unfinished_records(service):
    _pending_receipt(service)
    service._api = Mock(
        side_effect=AssertionError("Do not block the UI with a request")
    )
    assert service.jobs()[0]["status"] == "running"
    service._api.assert_not_called()


def test_process_identity_matches_live_process():
    import os
    from locaii.media import _process_identity

    assert _process_identity(os.getpid()).isdigit()
    assert _process_identity(999999999) == ""


def test_parent_guard_frees_idle_engine_without_interrupt(monkeypatch):
    import io
    import locaii.media as media

    requests = []
    monkeypatch.setattr(
        media, "_process_identity", lambda pid: "engine" if pid == 2 else ""
    )

    def request(value, timeout):
        requests.append(value)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(media, "urlopen", request)
    media._watch_parent(1, "parent", 2, "engine", "http://127.0.0.1:8188")
    assert requests[0].endswith("/queue")
    assert requests[1].full_url.endswith("/free")


def test_gpu_guard_refuses_foreign_work_without_stopping_it(service, monkeypatch):
    import locaii.media as media

    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    run = Mock(return_value=Mock(stdout="4321, foreign-model, 6000\n"))
    monkeypatch.setattr(media.subprocess, "run", run)
    service._is_configured_comfy_process = Mock(return_value=False)
    with pytest.raises(RuntimeError, match="GPU is busy"):
        MediaService._check_gpu_available(service)
    assert run.call_count == 1
    assert all("kill" not in str(call) for call in run.call_args_list)


def test_gpu_guard_reuses_only_idle_matching_engine(service, monkeypatch):
    import locaii.media as media

    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        media.subprocess, "run", Mock(return_value=Mock(stdout="4321, python, 6000\n"))
    )
    service._is_configured_comfy_process = Mock(return_value=True)
    service._api = Mock(side_effect=[{}, {}])
    MediaService._check_gpu_available(service)
    assert service._api.call_args_list[1].args == (
        "/free",
        {"unload_models": True, "free_memory": True},
    )


def test_gpu_guard_refuses_busy_matching_engine(service, monkeypatch):
    import locaii.media as media

    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        media.subprocess, "run", Mock(return_value=Mock(stdout="4321, python, 6000\n"))
    )
    service._is_configured_comfy_process = Mock(return_value=True)
    service._api = Mock(return_value={"queue_running": [[1, "foreign"]]})
    with pytest.raises(RuntimeError, match="busy with another job"):
        MediaService._check_gpu_available(service)
    assert service._api.call_count == 1


def test_gpu_guard_refuses_low_free_memory(service, monkeypatch):
    import locaii.media as media

    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        media.subprocess,
        "run",
        Mock(side_effect=[Mock(stdout=""), Mock(stdout="1024\n")]),
    )
    with pytest.raises(RuntimeError, match="only 1024"):
        MediaService._check_gpu_available(service)
