#!/usr/bin/env python3
"""Verify media installation or render bounded real image and video smoke tests.

Default mode is read-only and never starts ComfyUI. Pass --run only when the
GPU is idle. This script refuses foreign GPU workloads and never stops another
application. Receipts and outputs are retained under ignored local verification
state; media files remain in the configured ComfyUI output directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import socket
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.request import urlopen

from locaii.media import DEFAULTS, MediaService
from locaii.models import ModelManager

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_FILES = {
    "UnetLoaderGGUF": "custom_nodes/ComfyUI-GGUF/nodes.py",
    "Wan22ImageToVideoLatent": "comfy_extras/nodes_wan.py",
    "ModelSamplingSD3": "comfy_extras/nodes_model_advanced.py",
    "CreateVideo": "comfy_extras/nodes_video.py",
    "SaveVideo": "comfy_extras/nodes_video.py",
}
MODEL_INPUTS = {
    "ckpt_name": "checkpoints",
    "unet_name": "diffusion_models",
    "clip_name": "text_encoders",
    "vae_name": "vae",
}


def find_ffprobe() -> str | None:
    """Locate ffprobe when Python is launched without activating its conda env."""
    for candidate in (
        shutil.which("ffprobe"),
        str(Path(sys.executable).with_name("ffprobe")),
        str(Path(DEFAULTS["python"]).with_name("ffprobe")),
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def file_header(path: Path) -> dict:
    """Inspect only the weight header and metadata, without loading tensor bytes."""
    size = path.stat().st_size
    with path.open("rb") as source:
        leading = source.read(8)
        if path.suffix == ".gguf":
            valid = leading[:4] == b"GGUF" and size > 1024
            return {"format": "GGUF", "header_valid": valid, "bytes": size}
        header_size = int.from_bytes(leading, "little")
        if not 2 <= header_size <= 32 * 1024 * 1024 or header_size + 8 >= size:
            return {"format": "safetensors", "header_valid": False, "bytes": size}
        header = json.loads(source.read(header_size))
        tensors = [value for key, value in header.items() if key != "__metadata__"]
        valid = bool(tensors) and all(
            isinstance(item, dict)
            and "data_offsets" in item
            and 0
            <= item["data_offsets"][0]
            <= item["data_offsets"][1]
            <= size - header_size - 8
            for item in tensors
        )
        return {
            "format": "safetensors",
            "header_valid": valid,
            "bytes": size,
            "tensor_count": len(tensors),
        }


def model_directories(service: MediaService) -> dict[str, list[Path]]:
    """Read ComfyUI YAML through its existing PyYAML, without importing torch."""
    comfy = Path(service.config["comfy_dir"])
    result = {kind: [comfy / "models" / kind] for kind in MODEL_INPUTS.values()}
    config = comfy / "extra_model_paths.yaml"
    if not config.exists():
        return result
    completed = subprocess.run(
        [
            service.config["python"],
            "-c",
            "import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))",
            str(config),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    for entry in json.loads(completed.stdout).values():
        base = Path(entry.get("base_path", str(comfy)))
        if not base.is_absolute():
            base = comfy / base
        for kind in result:
            for folder in str(entry.get(kind, "")).splitlines():
                if folder.strip():
                    result[kind].append((base / folder.strip()).resolve())
    return result


def validate_installation(service: MediaService) -> dict:
    """Check source-defined graph nodes and actual required weight file headers."""
    comfy = Path(service.config["comfy_dir"])
    directories = model_directories(service)
    report = {
        "mode": "static",
        "engine": str(comfy),
        "nodes": [],
        "weights": [],
        "passed": True,
        "limitations": "Source/header checks do not prove runtime node imports, rendering, or content quality.",
    }
    classes = {}
    weights_seen = set()
    nodes_seen = set()
    for workflow in service.list_workflows()[:2]:
        graph = json.loads(Path(workflow["path"]).read_text())
        for node in graph.values():
            name = node["class_type"]
            source = comfy / SOURCE_FILES.get(name, "nodes.py")
            if source not in classes:
                tree = ast.parse(source.read_text())
                classes[source] = {
                    item.name
                    for item in ast.walk(tree)
                    if isinstance(item, ast.ClassDef)
                }
            if name not in nodes_seen:
                item = {
                    "name": name,
                    "source": str(source),
                    "present": name in classes[source],
                }
                report["nodes"].append(item)
                report["passed"] &= item["present"]
                nodes_seen.add(name)
            for key, kind in MODEL_INPUTS.items():
                name = node["inputs"].get(key)
                if not name or name in weights_seen:
                    continue
                candidates = [directory / name for directory in directories[kind]]
                path = next((path for path in candidates if path.is_file()), None)
                item = {
                    "name": name,
                    "category": kind,
                    "path": str(path) if path else None,
                    "present": path is not None,
                }
                if path:
                    item.update(file_header(path))
                report["weights"].append(item)
                report["passed"] &= item["present"] and item.get("header_valid", False)
                weights_seen.add(name)
    report["ffprobe"] = find_ffprobe()
    report["passed"] &= bool(report["ffprobe"])
    return report


def require_idle_gpu() -> dict:
    """Refuse every foreign compute process and insufficient free GPU memory."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise RuntimeError("Cannot verify GPU ownership: nvidia-smi is unavailable.")
    apps = subprocess.run(
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
    processes = list(csv.reader(io.StringIO(apps.stdout)))
    processes = [row for row in processes if row and row[0].strip()]
    if processes:
        # Names and PIDs suffice. Never inspect command lines or API secrets.
        summary = ", ".join(
            f"PID {row[0].strip()} ({row[1].strip()})" for row in processes
        )
        raise RuntimeError(
            "GPU smoke test refused: existing compute workload: " + summary
        )
    stats = subprocess.run(
        [
            executable,
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    first = next(csv.reader(io.StringIO(stats.stdout)))
    free, total = map(lambda item: int(item.strip()), first)
    if free < 6500:
        raise RuntimeError(
            f"GPU smoke test refused: only {free} MiB free; at least 6500 MiB is required."
        )
    return {"free_mib": free, "total_mib": total, "compute_processes": []}


def verify_image(path: Path, width: int, height: int) -> dict:
    """Decode and verify actual pixels, dimensions and complete output bytes."""
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        result = {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "passed": image.size == (width, height),
        }
    result.update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return result


def verify_video(path: Path, width: int, height: int, frames: int) -> dict:
    """Ask ffprobe to decode/count actual video frames, not just trust its filename."""
    completed = subprocess.run(
        [
            find_ffprobe() or "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    result = streams[0] if streams else {}
    result["passed"] = (
        result.get("width") == width
        and result.get("height") == height
        and int(result.get("nb_read_frames", 0)) == frames
    )
    result.update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return result


def check_cpu_api(service: MediaService, directory: Path) -> dict:
    """Start an owned CPU-only engine and inspect schemas without submitting work."""
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    available = int(memory["MemAvailable"].split()[0]) * 1024
    if available < 4 * 1024**3:
        raise RuntimeError("CPU API validation needs at least 4 GiB available RAM.")
    # Refuse an occupied port before launch. Never adopt or stop that process.
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 18188)) == 0:
            raise RuntimeError("CPU API test port 18188 is already in use.")
    comfy = Path(service.config["comfy_dir"])
    command = [
        service.config["python"],
        "main.py",
        "--cpu",
        "--listen",
        "127.0.0.1",
        "--port",
        "18188",
        "--output-directory",
        str(directory / "unused-output"),
        "--disable-api-nodes",
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        "ComfyUI-GGUF",
        "local_media_cpu_clip.py",
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    started = time.monotonic()
    report = {
        "mode": "cpu_api_schema_only",
        "port": 18188,
        "device": "cpu",
        "prompts_submitted": 0,
        "weights_loaded": False,
        "available_ram_bytes": available,
        "passed": False,
        "checks": [],
    }
    with (directory / "cpu-comfy.log").open("wb") as log:
        process = subprocess.Popen(
            command, cwd=comfy, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("CPU ComfyUI exited; inspect cpu-comfy.log.")
                try:
                    with urlopen(
                        "http://127.0.0.1:18188/system_stats", timeout=2
                    ) as response:
                        report["system_stats"] = json.load(response)
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                raise TimeoutError("CPU ComfyUI readiness exceeded 180 seconds.")
            with urlopen("http://127.0.0.1:18188/object_info", timeout=10) as response:
                schema = json.load(response)
            required_names = set()
            for workflow in service.list_workflows()[:2]:
                graph = json.loads(Path(workflow["path"]).read_text())
                for node_id, node in graph.items():
                    name = node["class_type"]
                    required_names.add(name)
                    definition = schema.get(name)
                    errors = []
                    if definition is None:
                        errors.append("Node missing from live engine registry")
                    else:
                        required = definition.get("input", {}).get("required", {})
                        optional = definition.get("input", {}).get("optional", {})
                        for key in required:
                            if key not in node["inputs"]:
                                errors.append("Missing required input: " + key)
                        for key, value in node["inputs"].items():
                            entry = required.get(key, optional.get(key))
                            # Dynamic codec format inputs may be represented by
                            # their parent schema rather than a top-level field.
                            if entry is None:
                                if "." not in key:
                                    errors.append("Unknown input: " + key)
                                continue
                            if (
                                isinstance(value, list)
                                and len(value) == 2
                                and str(value[0]) in graph
                            ):
                                upstream = schema.get(
                                    graph[str(value[0])]["class_type"], {}
                                )
                                outputs = upstream.get("output", [])
                                if not isinstance(value[1], int) or not 0 <= value[
                                    1
                                ] < len(outputs):
                                    errors.append("Invalid linked output: " + key)
                            elif isinstance(entry[0], list) and value not in entry[0]:
                                errors.append(
                                    "Input value is absent from live choices: " + key
                                )
                            elif (
                                len(entry) > 1
                                and isinstance(entry[1], dict)
                                and isinstance(value, (int, float))
                            ):
                                options = entry[1]
                                if value < options.get(
                                    "min", value
                                ) or value > options.get("max", value):
                                    errors.append(
                                        "Input value outside live bounds: " + key
                                    )
                    report["checks"].append(
                        {
                            "workflow": workflow["name"],
                            "node_id": node_id,
                            "node": name,
                            "passed": not errors,
                            "errors": errors,
                        }
                    )
            selected = {name: schema.get(name) for name in sorted(required_names)}
            (directory / "live-node-schemas.json").write_text(
                json.dumps(selected, indent=2)
            )
            with urlopen("http://127.0.0.1:18188/queue", timeout=2) as response:
                queue = json.load(response)
            report["queue_empty"] = not queue.get("queue_running") and not queue.get(
                "queue_pending"
            )
            report["cpu_device_confirmed"] = bool(
                report["system_stats"].get("devices")
            ) and all(
                device.get("type") == "cpu"
                for device in report["system_stats"]["devices"]
            )
            report["passed"] = (
                report["queue_empty"]
                and report["cpu_device_confirmed"]
                and all(row["passed"] for row in report["checks"])
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            report["owned_process_stopped"] = process.poll() is not None
            report["seconds"] = time.monotonic() - started
            (directory / "cpu-api-report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    """Persist installation findings and, when explicitly requested, real receipts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-api",
        action="store_true",
        help="Start CPU-only engine for schemas, never render",
    )
    parser.add_argument(
        "--run", action="store_true", help="Render only after verifying the GPU is idle"
    )
    parser.add_argument("--kind", choices=["both", "image", "video"], default="both")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--video-width", type=int, default=832)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-frames", type=int, default=33)
    parser.add_argument(
        "--legacy-memory",
        action="store_true",
        help="Test ComfyUI with dynamic VRAM disabled",
    )
    parser.add_argument(
        "--video-steps",
        type=int,
        default=20,
        help="Wan sampling steps; 20 matches the official base workflow",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / ".local/verification/media"
    )
    args = parser.parse_args()
    if args.cpu_api and args.run:
        parser.error("Use --cpu-api or --run, not both.")
    root = args.output.resolve() / uuid.uuid4().hex
    root.mkdir(parents=True)
    # A new manager owns no foreign process. MediaService.stop() can affect only
    # a process created by this test manager, never an existing application's model.
    if args.legacy_memory:
        (root / "config.json").write_text(
            json.dumps({"media": {"disable_dynamic_vram": True}})
        )
    manager = ModelManager(root / "config.json", root / "chat-runtime")
    service = MediaService(root / "config.json", root / "state", manager)
    report = {
        "started_at": time.time(),
        "run_requested": args.run,
        "passed": False,
        "results": [],
        "directory": str(root),
    }
    cancel = threading.Event()
    timer = threading.Timer(args.timeout, cancel.set)

    def emit(event, payload):
        record = {"at": time.time(), "event": event, **payload}
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    try:
        report["installation"] = validate_installation(service)
        if not report["installation"]["passed"]:
            raise RuntimeError("Static installation check failed.")
        if not args.run:
            report["passed"] = True
            report["generation_status"] = "not_run"
            if args.cpu_api:
                report["cpu_api"] = check_cpu_api(service, root)
                report["passed"] = report["cpu_api"]["passed"]
        else:
            report["gpu_preflight"] = require_idle_gpu()
            timer.start()
            tasks = [
                (
                    "image",
                    "A small red ceramic teapot on a wooden table, soft window light, realistic product photo.",
                    {"width": 512, "height": 512, "steps": 5, "seed": 42},
                ),
                (
                    "video",
                    "A small red sailboat glides slowly across a calm blue lake, gentle ripples, fixed camera.",
                    {
                        "width": args.video_width,
                        "height": args.video_height,
                        "frames": args.video_frames,
                        "steps": args.video_steps,
                        "fps": 24,
                        "seed": 42,
                    },
                ),
            ]
            for kind, prompt, params in tasks:
                if args.kind not in {"both", kind}:
                    continue
                # Recheck between tests so another application is never displaced.
                require_idle_gpu()
                receipt = service.generate(
                    kind, prompt, {**params, "timeout": args.timeout}, cancel, emit
                )
                suffixes = (
                    {".png", ".jpg", ".webp"} if kind == "image" else {".mp4", ".webm"}
                )
                path = next(
                    Path(path)
                    for path in receipt["files"]
                    if Path(path).suffix.lower() in suffixes
                )
                verified = (
                    verify_image(path, params["width"], params["height"])
                    if kind == "image"
                    else verify_video(
                        path, params["width"], params["height"], params["frames"]
                    )
                )
                report["results"].append(
                    {"kind": kind, "receipt": receipt, "verification": verified}
                )
                if not verified["passed"]:
                    raise RuntimeError(
                        f"The generated {kind} did not meet its file verification checks."
                    )
            report["passed"] = bool(report["results"])
            report["generation_status"] = "completed"
    except Exception as error:
        report["error"] = str(error)
        report["generation_status"] = "refused_or_failed"
    finally:
        cancel.set()
        timer.cancel()
        manager.stop()
        report["finished_at"] = time.time()
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
