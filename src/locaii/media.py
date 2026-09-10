"""Standalone ComfyUI adapter with durable jobs and exclusive local GPU use.

All generation happens in an external ComfyUI Python environment. No chat UI,
LM Studio process, or proprietary media API is needed. API workflow JSON files
are the extension interface. Imported node graphs can execute installed custom
nodes, so users should only import workflows they trust.
"""

from __future__ import annotations

import copy
import csv
import io
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "comfy_dir": "/home/andry/.local/opt/ComfyUI",
    "python": "/home/andry/miniconda3/envs/local-media/bin/python",
    "url": "http://127.0.0.1:8188",
    "output_dir": "/run/media/andry/andryData/AI_Generated",
    "startup_timeout": 180,
    "generation_timeout": 3600,
    "poll_seconds": 1,
    "min_free_vram_mib": 6500,
}
BUILTINS = [
    {
        "name": "RealVisXL Lightning",
        "kind": "image",
        "path": str(ROOT / "templates/image-api.json"),
        "mapping": {
            "prompt": ["4", "text"],
            "negative_prompt": ["5", "text"],
            "width": ["9", "width"],
            "height": ["9", "height"],
            "seed": ["10", "seed"],
            "steps": ["10", "steps"],
        },
    },
    {
        "name": "Wan 2.2 5B",
        "kind": "video",
        "path": str(ROOT / "templates/wan-api.json"),
        "mapping": {
            "prompt": ["3", "text"],
            "negative_prompt": ["4", "text"],
            "width": ["5", "width"],
            "height": ["5", "height"],
            "frames": ["5", "length"],
            "seed": ["12", "seed"],
            "steps": ["12", "steps"],
            "fps": ["57", "fps"],
        },
        "image_input_node": "5",
    },
]


class MediaCancelled(RuntimeError):
    """The caller cancelled only its own generation."""


def _atomic_json(path: Path, value: Any) -> None:
    """Publish a complete UTF-8 record after flushing it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class MediaService:
    """Render images/video while keeping ownership, receipts and errors explicit."""

    def __init__(self, config_path: Path, state_dir: Path, model_manager: Any):
        self.config_path = Path(config_path)
        self.state_dir = Path(state_dir) / "media"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.model_manager = model_manager
        self.config = dict(DEFAULTS)
        if self.config_path.exists():
            self.config.update(
                json.loads(self.config_path.read_text()).get("media", {})
            )
        parsed = urlparse(self.config["url"])
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("ComfyUI must use a loopback HTTP URL.")
        self._process: subprocess.Popen | None = None
        self._log = None
        self._lock = threading.Lock()

    def _api(self, route: str, payload: Any = None, *, timeout: float = 10) -> Any:
        """Call the local API with bounded request time and useful error details."""
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            self.config["url"].rstrip("/") + route,
            body,
            {"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(16000).decode(errors="replace")
            raise RuntimeError(f"ComfyUI HTTP {error.code}: {detail}") from error

    def list_workflows(self) -> list[dict]:
        """Return built-in and user-registered API workflows with capabilities."""
        entries = copy.deepcopy(BUILTINS)
        registry = self.state_dir / "workflows.json"
        if registry.exists():
            entries.extend(json.loads(registry.read_text()))
        entries.extend(copy.deepcopy(self.config.get("workflows", [])))
        for item in entries:
            item["supports_image_input"] = bool(item.get("image_input_node"))
            item["available"] = Path(item["path"]).is_file()
        return entries

    @staticmethod
    def _discover_mapping(graph: dict) -> dict:
        """Infer common controls from API nodes, requiring unambiguous positive text."""
        result = {}
        positive = []
        for node in graph.values():
            connection = node.get("inputs", {}).get("positive")
            if (
                isinstance(connection, list)
                and connection
                and str(connection[0]) in graph
            ):
                positive.append(str(connection[0]))
        text_nodes = [
            key
            for key, node in graph.items()
            if isinstance(node.get("inputs", {}).get("text"), str)
        ]
        prompt_nodes = list(dict.fromkeys(positive)) or text_nodes
        if len(prompt_nodes) != 1:
            raise ValueError(
                "Cannot identify one positive prompt node. Import a wrapper with an explicit mapping."
            )
        result["prompt"] = [prompt_nodes[0], "text"]
        for key in ("width", "height", "seed", "steps", "fps"):
            candidates = [
                node_id
                for node_id, node in graph.items()
                if isinstance(node.get("inputs", {}).get(key), (int, float))
            ]
            if len(candidates) == 1:
                result[key] = [candidates[0], key]
        lengths = [
            key
            for key, node in graph.items()
            if isinstance(node.get("inputs", {}).get("length"), int)
        ]
        if len(lengths) == 1:
            result["frames"] = [lengths[0], "length"]
        return result

    def register_workflow(self, name: str, path: Path, kind: str) -> dict:
        """Copy a trusted API graph, optionally wrapped with explicit control mapping."""
        if kind not in {"image", "video"} or not name.strip():
            raise ValueError("Provide a workflow name and image/video kind.")
        source = Path(path)
        if source.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Workflow JSON exceeds 8 MiB.")
        loaded = json.loads(source.read_text())
        graph = loaded.get("graph", loaded)
        if (
            not isinstance(graph, dict)
            or not graph
            or any(
                not isinstance(node, dict)
                or "class_type" not in node
                or "inputs" not in node
                for node in graph.values()
            )
        ):
            raise ValueError(
                "Export ComfyUI API format, not its visual editor workflow format."
            )
        mapping = loaded.get("mapping") or self._discover_mapping(graph)
        if "prompt" not in mapping:
            raise ValueError("Workflow mapping must contain prompt.")
        for control, location in mapping.items():
            if (
                not isinstance(location, list)
                or len(location) != 2
                or location[0] not in graph
                or location[1] not in graph[location[0]]["inputs"]
            ):
                raise ValueError(f"Invalid workflow control mapping: {control}")
        destination = self.state_dir / "workflows" / (uuid.uuid4().hex + ".json")
        _atomic_json(destination, graph)
        entry = {
            "name": name.strip(),
            "kind": kind,
            "path": str(destination),
            "mapping": mapping,
        }
        image_node = loaded.get("image_input_node")
        if image_node is not None:
            if (
                str(image_node) not in graph
                or graph[str(image_node)]["class_type"] != "Wan22ImageToVideoLatent"
            ):
                raise ValueError(
                    "Image input currently supports verified Wan22ImageToVideoLatent only."
                )
            entry["image_input_node"] = str(image_node)
        registry = self.state_dir / "workflows.json"
        with self._lock:
            entries = json.loads(registry.read_text()) if registry.exists() else []
            if any(item["name"] == entry["name"] for item in self.list_workflows()):
                raise ValueError("A workflow with this name already exists.")
            entries.append(entry)
            _atomic_json(registry, entries)
        return entry

    def jobs(self) -> list[dict]:
        """Read newest-first durable receipts, including interrupted previous sessions."""
        records = []
        for path in (self.state_dir / "jobs").glob("*.json"):
            if path.name.endswith(".history.json"):
                continue
            try:
                records.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda row: row.get("started_at", 0), reverse=True)

    def recover_jobs(self) -> list[dict]:
        """Reconcile saved jobs in a background worker using two bounded API reads.

        History and queue are queried in batches, never once per receipt. A
        transient timeout preserves uncertainty. A refused loopback connection
        establishes that this engine is unavailable, but cannot establish whether
        a previous job completed before shutdown, so its outcome is interrupted.
        """
        records = self.jobs()
        pending = [
            row
            for row in records
            if row.get("status")
            in {"starting", "running", "queued", "unknown", "interrupted"}
        ]
        if not pending:
            return records
        if not self._lock.acquire(blocking=False):
            return records
        generation_lock = (self.state_dir / "generation.lock").open("a")
        try:
            try:
                fcntl.flock(generation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Another live app owns these records and will write its receipt.
                return records
            pending = [
                row
                for row in self.jobs()
                if row.get("status")
                in {"starting", "running", "queued", "unknown", "interrupted"}
            ]
            if not pending:
                return self.jobs()
            try:
                queue = self._api("/queue", timeout=2)
                history = self._api("/history?max_items=200", timeout=2)
            except Exception as error:
                cause = error.reason if isinstance(error, URLError) else error
                refused = isinstance(cause, ConnectionRefusedError)
                for row in pending:
                    row.update(
                        status="interrupted" if refused else "unknown",
                        recovery_error=(
                            "Engine is unavailable; previous outcome is unconfirmed."
                            if refused
                            else "Engine status could not be checked: " + str(error)
                        ),
                        recovered_at=time.time(),
                    )
                    _atomic_json(
                        self.state_dir / "jobs" / (row["job_id"] + ".json"), row
                    )
                return self.jobs()
            active_ids = {
                item[1]
                for group in ("queue_running", "queue_pending")
                for item in queue.get(group, [])
            }
            for row in pending:
                prompt_id = row.get("prompt_id")
                result = history.get(prompt_id) if prompt_id else None
                row["recovered_at"] = time.time()
                row.pop("recovery_error", None)
                if result is not None:
                    _atomic_json(
                        self.state_dir / "jobs" / (row["job_id"] + ".history.json"),
                        result,
                    )
                    try:
                        if result.get("status", {}).get("status_str") != "success":
                            raise RuntimeError(
                                "Generation failed: "
                                + json.dumps(result.get("status", {}))[:8000]
                            )
                        row.update(status="completed", files=self._files(result))
                        row.pop("error", None)
                    except Exception as error:
                        row.update(status="failed", error=str(error))
                elif prompt_id in active_ids:
                    row["status"] = "running"
                else:
                    row.update(
                        status="interrupted",
                        recovery_error="Prompt is absent from the current queue and recent engine history. "
                        "Its prior output is unconfirmed; it was not automatically resubmitted.",
                    )
                _atomic_json(self.state_dir / "jobs" / (row["job_id"] + ".json"), row)
            # A surviving engine from a crashed GUI can now release idle models.
            # This schedules memory release rather than terminating shared work.
            if not active_ids:
                try:
                    self._api(
                        "/free", {"unload_models": True, "free_memory": True}, timeout=2
                    )
                except Exception:
                    pass
            return self.jobs()
        finally:
            fcntl.flock(generation_lock, fcntl.LOCK_UN)
            generation_lock.close()
            self._lock.release()

    def _is_configured_comfy_process(self, pid: int) -> bool:
        """Recognize only the configured local engine, without displaying argv."""
        try:
            directory = Path(self.config["comfy_dir"]).resolve()
            if Path(f"/proc/{pid}/cwd").resolve() != directory:
                return False
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            if not any(
                item == "main.py" or item == str(directory / "main.py") for item in argv
            ):
                return False
            expected = str(urlparse(self.config["url"]).port or 8188)
            actual = argv[argv.index("--port") + 1] if "--port" in argv else "8188"
            return actual == expected
        except (OSError, ValueError, IndexError):
            return False

    def _check_gpu_available(self) -> None:
        """Fail visibly before loading media into foreign occupied GPU memory.

        The already released chat manager is the only model we may stop. An
        existing matching ComfyUI process can be reused only when its queue is
        idle. No other compute process is terminated or unloaded.
        """
        executable = shutil.which("nvidia-smi")
        if not executable:
            raise RuntimeError(
                "Cannot verify GPU availability: nvidia-smi is unavailable."
            )
        checked = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        foreign = []
        own_engine = False
        for row in csv.reader(io.StringIO(checked.stdout)):
            if not row or not row[0].strip():
                continue
            pid = int(row[0].strip())
            if self._is_configured_comfy_process(pid):
                own_engine = True
            else:
                foreign.append(pid)
        if foreign:
            raise RuntimeError(
                "GPU is busy with another application (process IDs "
                + ", ".join(map(str, foreign))
                + "). Let its task finish, then release "
                "its model before generating media. loCAII did not stop it."
            )
        if own_engine:
            queue = self._api("/queue", timeout=2)
            if queue.get("queue_running") or queue.get("queue_pending"):
                raise RuntimeError(
                    "The media engine is busy with another job. Let it finish first."
                )
            # A reusable idle engine can unload cached weights before it renders
            # the next workflow. Its scheduler performs this only between jobs.
            self._api("/free", {"unload_models": True, "free_memory": True}, timeout=2)
            return
        checked = subprocess.run(
            [executable, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        free = int(checked.stdout.splitlines()[0].strip())
        minimum = int(self.config["min_free_vram_mib"])
        if free < minimum:
            raise RuntimeError(
                f"GPU has only {free} MiB free; media requires at least {minimum} MiB. "
                "Close another GPU application before trying again."
            )

    def _ensure_service(self, cancel: threading.Event, emit: Callable) -> None:
        """Reuse an available local server or launch ComfyUI directly in its environment."""
        try:
            self._api("/system_stats")
            return
        except (OSError, ValueError, RuntimeError):
            pass
        directory = Path(self.config["comfy_dir"])
        executable = Path(self.config["python"])
        if not (directory / "main.py").is_file() or not executable.is_file():
            raise RuntimeError(
                "ComfyUI is not installed at the configured path. See docs/MEDIA.md."
            )
        output = Path(self.config["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        port = str(urlparse(self.config["url"]).port or 8188)
        args = [
            str(executable),
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            port,
            "--output-directory",
            str(output),
            "--mmap-torch-files",
            "--reserve-vram",
            "1",
            "--disable-api-nodes",
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes",
            "ComfyUI-GGUF",
            "local_media_cpu_clip.py",
        ]
        self._log = (self.state_dir / "comfy.log").open("ab")
        environment = dict(
            os.environ,
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        self._process = subprocess.Popen(
            args,
            cwd=directory,
            env=environment,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        # A lightweight detached watcher lets rendering finish if the GUI
        # crashes, then unloads idle models while leaving history recoverable.
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--watch-parent",
                str(os.getpid()),
                _process_identity(os.getpid()),
                str(self._process.pid),
                _process_identity(self._process.pid),
                self.config["url"],
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        deadline = time.monotonic() + float(self.config["startup_timeout"])
        emit(
            "status",
            {
                "stage": "starting",
                "message": "Starting the local image and video engine",
            },
        )
        while time.monotonic() < deadline:
            if cancel.is_set():
                raise MediaCancelled("Generation cancelled before submission.")
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited. Inspect {self.state_dir / 'comfy.log'}"
                )
            try:
                self._api("/system_stats")
                return
            except (OSError, ValueError, RuntimeError):
                cancel.wait(0.5)
        raise TimeoutError("ComfyUI startup timed out.")

    def prepare_graph(
        self, workflow: dict, prompt: str, params: dict, job_id: str
    ) -> dict:
        """Apply validated controls to a fresh graph and optional verified Wan input."""
        graph = json.loads(Path(workflow["path"]).read_text())
        controls = {
            "prompt": prompt.strip(),
            **{
                key: value
                for key, value in params.items()
                if key in workflow["mapping"]
            },
        }
        if not controls["prompt"]:
            raise ValueError("Enter a generation prompt.")
        for key in ("width", "height", "steps", "frames", "fps", "seed"):
            if key not in controls:
                continue
            value = int(controls[key])
            if key in {"width", "height"} and (
                value < 64 or value > 2048 or value % 32
            ):
                raise ValueError(
                    "Width and height must be multiples of 32 between 64 and 2048."
                )
            if key == "steps" and not 1 <= value <= 100:
                raise ValueError("Steps must be between 1 and 100.")
            if key == "frames" and (not 9 <= value <= 129 or (value - 1) % 8):
                raise ValueError("Video frames must be 8k + 1, from 9 to 129.")
            if key == "fps" and not 1 <= value <= 60:
                raise ValueError("FPS must be between 1 and 60.")
            if key == "seed" and not 0 <= value < 2**63:
                raise ValueError("Seed must be a nonnegative 63-bit integer.")
            controls[key] = value
        for control, value in controls.items():
            node, key = workflow["mapping"][control]
            graph[node]["inputs"][key] = value
        # Give every job its own output prefix to make provenance unambiguous.
        for node in graph.values():
            if "filename_prefix" in node["inputs"]:
                node["inputs"][
                    "filename_prefix"
                ] = f"loCAII/{workflow['kind']}/{job_id}"
        image_path = params.get("image_path") or params.get("input_image")
        if image_path:
            latent = workflow.get("image_input_node")
            if not latent or graph[latent]["class_type"] != "Wan22ImageToVideoLatent":
                raise ValueError(
                    "This workflow does not support an uploaded source image."
                )
            source = Path(image_path).expanduser().resolve()
            if not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
                raise ValueError(
                    "Source image must be an existing file smaller than 32 MiB."
                )
            from PIL import Image

            with Image.open(source) as opened:
                opened.verify()
            name = job_id + source.suffix.lower()
            target = Path(self.config["comfy_dir"]) / "input" / "locaii" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            node_id = "locaii_source_image"
            graph[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": "locaii/" + name},
            }
            graph[latent]["inputs"]["start_image"] = [node_id, 0]
        return graph

    def _cancel_prompt(self, prompt_id: str) -> None:
        """Use atomic ID-scoped cancellation. Never issue a global interrupt."""
        try:
            self._api(f"/api/jobs/{prompt_id}/cancel", {})
        except Exception as error:
            # A queued prompt can safely be removed by ID on older engines. An
            # unowned active engine is never killed or globally interrupted.
            self._api("/queue", {"delete": [prompt_id]})
            if self._process is None:
                raise RuntimeError(
                    "Engine does not support safe active-job cancellation; job may still be running."
                ) from error

    def _files(self, history: dict) -> list[str]:
        """Accept existing output files strictly inside the configured output root."""
        root = Path(self.config["output_dir"]).resolve()
        files = []
        for output in history.get("outputs", {}).values():
            for values in output.values():
                if not isinstance(values, list):
                    continue
                for item in values:
                    if (
                        not isinstance(item, dict)
                        or item.get("type") != "output"
                        or "filename" not in item
                    ):
                        continue
                    path = (
                        root / item.get("subfolder", "") / item["filename"]
                    ).resolve()
                    if not path.is_relative_to(root):
                        raise RuntimeError(
                            "ComfyUI returned an output outside its configured directory."
                        )
                    if path.is_file() and path.stat().st_size:
                        files.append(str(path))
        if not files:
            raise RuntimeError(
                "Generation reported success but no nonempty output file was found."
            )
        return list(dict.fromkeys(files))

    def _release(self, owned_prompt: str | None = None) -> None:
        """Release only our child process, or request idle shared-engine model unload."""
        if self._process is not None:
            try:
                queue = self._api("/queue")
                foreign = [
                    item
                    for group in ("queue_running", "queue_pending")
                    for item in queue.get(group, [])
                    if item[1] != owned_prompt
                ]
                if foreign:
                    # Someone used our loopback server while we rendered. Leave
                    # it alive rather than terminating their active or queued work.
                    self._process = None
                    if self._log:
                        self._log.close()
                        self._log = None
                    return
            except Exception:
                pass
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._process = None
        else:
            try:
                queue = self._api("/queue")
                if not queue.get("queue_running") and not queue.get("queue_pending"):
                    self._api("/free", {"unload_models": True, "free_memory": True})
            except Exception:
                pass
        if self._log:
            self._log.close()
            self._log = None

    def generate(
        self,
        kind: str,
        prompt: str,
        params: dict,
        cancel: threading.Event,
        emit: Callable[[str, dict], None],
    ) -> dict:
        """Run one durable generation and emit observed stages, elapsed time and receipts.

        Progress is based on the actual queue/history. No invented completion
        percentages are emitted while the engine loads or samples its model.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A media job is already running in this application.")
        job_id = uuid.uuid4().hex
        record = {
            "job_id": job_id,
            "kind": kind,
            "prompt": prompt,
            "params": params,
            "started_at": time.time(),
            "status": "starting",
            "files": [],
        }
        path = self.state_dir / "jobs" / (job_id + ".json")
        process_lock = (self.state_dir / "generation.lock").open("a")
        started = time.monotonic()
        service_acquired = False
        try:
            try:
                fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "Another loCAII instance is generating media."
                ) from error
            _atomic_json(path, record)
            choices = [item for item in self.list_workflows() if item["kind"] == kind]
            name = params.get("workflow")
            workflow = next(
                (item for item in choices if name is None or item["name"] == name), None
            )
            if workflow is None:
                raise ValueError("No matching image/video workflow is registered.")
            graph = self.prepare_graph(workflow, prompt, params, job_id)
            if cancel.is_set():
                raise MediaCancelled("Generation cancelled before submission.")
            # The manager owns inference synchronization; stop refuses active work.
            self.model_manager.stop()
            self._check_gpu_available()
            self._ensure_service(cancel, emit)
            service_acquired = True
            queue = self._api("/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                raise RuntimeError(
                    "ComfyUI is busy with another job. Wait for it to finish."
                )
            if cancel.is_set():
                raise MediaCancelled("Generation cancelled before submission.")
            submitted = self._api("/prompt", {"prompt": graph, "client_id": job_id})
            prompt_id = submitted.get("prompt_id")
            if not prompt_id or submitted.get("node_errors"):
                raise RuntimeError("Workflow rejected: " + json.dumps(submitted))
            record.update(
                status="running", prompt_id=prompt_id, workflow=workflow["name"]
            )
            _atomic_json(path, record)
            deadline = started + float(
                params.get("timeout", self.config["generation_timeout"])
            )
            while time.monotonic() < deadline:
                if cancel.is_set():
                    self._cancel_prompt(prompt_id)
                    raise MediaCancelled("Generation cancelled.")
                history = self._api("/history/" + prompt_id)
                if prompt_id in history:
                    result = history[prompt_id]
                    _atomic_json(path.with_suffix(".history.json"), result)
                    if result.get("status", {}).get("status_str") != "success":
                        raise RuntimeError(
                            "Generation failed: "
                            + json.dumps(result.get("status", {}))[:8000]
                        )
                    files = self._files(result)
                    record.update(
                        status="completed",
                        files=files,
                        seconds=time.monotonic() - started,
                    )
                    _atomic_json(path, record)
                    emit("completed", dict(record))
                    return record
                queue = self._api("/queue")
                running = any(
                    item[1] == prompt_id for item in queue.get("queue_running", [])
                )
                emit(
                    "progress",
                    {
                        "job_id": job_id,
                        "stage": "running" if running else "queued",
                        "seconds": time.monotonic() - started,
                        "message": (
                            "Engine is rendering"
                            if running
                            else "Waiting in engine queue"
                        ),
                    },
                )
                cancel.wait(float(self.config["poll_seconds"]))
            self._cancel_prompt(prompt_id)
            raise TimeoutError("Media generation reached its configured time limit.")
        except Exception as error:
            if record.get("prompt_id") and record["status"] == "running":
                try:
                    self._cancel_prompt(record["prompt_id"])
                except Exception as cancel_error:
                    record["cancellation_error"] = str(cancel_error)
            record.update(
                status="cancelled" if isinstance(error, MediaCancelled) else "failed",
                error=str(error),
                seconds=time.monotonic() - started,
            )
            _atomic_json(path, record)
            emit("error", dict(record))
            raise
        finally:
            if service_acquired or self._process is not None:
                self._release(record.get("prompt_id"))
            fcntl.flock(process_lock, fcntl.LOCK_UN)
            process_lock.close()
            self._lock.release()


def _process_identity(pid: int) -> str:
    """Identify a Linux process by start tick, avoiding PID reuse mistakes."""
    try:
        # The comm field may itself contain spaces or parentheses.
        value = Path(f"/proc/{pid}/stat").read_text()
        fields = value[value.rfind(")") + 2 :].split()
        return "" if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return ""


def _watch_parent(
    parent_pid: int,
    parent_identity: str,
    engine_pid: int,
    engine_identity: str,
    url: str,
) -> None:
    """After a GUI crash, release idle GPU models without killing any work.

    The guard uses only process identity and loopback queue/free APIs. It does
    not restart servers, inspect prompts, cancel jobs, or delete engine history.
    """
    while engine_identity and _process_identity(engine_pid) == engine_identity:
        if parent_identity and _process_identity(parent_pid) == parent_identity:
            time.sleep(2)
            continue
        try:
            with urlopen(url.rstrip("/") + "/queue", timeout=2) as response:
                queue = json.load(response)
            if not queue.get("queue_running") and not queue.get("queue_pending"):
                request = Request(
                    url.rstrip("/") + "/free",
                    json.dumps({"unload_models": True, "free_memory": True}).encode(),
                    {"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=2) as response:
                    response.read(1000)
                return
        except Exception:
            pass
        time.sleep(5)


if __name__ == "__main__" and len(sys.argv) == 7 and sys.argv[1] == "--watch-parent":
    _watch_parent(
        int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
    )
