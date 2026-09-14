"""Owned llama.cpp process, editable model registry, and streamed native tool calls.

All prediction and lifecycle operations serialize through one reentrant lock.
Cancellation closes the HTTP request; only the child process created here can be
terminated. Other model servers, including LM Studio, are never stopped here.
"""

from __future__ import annotations
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import secrets
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
import requests

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path("/run/media/andry/andryData/LLM_Models")


class ModelError(RuntimeError):
    """An actionable local runtime or inference error."""


def _backend_error_detail(value) -> str:
    """Expose the cause of a local backend failure without echoing prompt data.

    llama.cpp can include a complete, malformed tool argument in its exception.
    For JSON parsing errors, retain only the exception code, numeric location,
    and a recognized parser diagnosis. Other errors receive a short sanitized
    summary with quoted strings, serialized payloads and credentials removed.
    """
    if isinstance(value, dict):
        value = value.get("error", value)
        if isinstance(value, dict):
            value = (
                value.get("message")
                or value.get("detail")
                or value.get("type")
                or "Backend rejected the request."
            )
    if not isinstance(value, str):
        return "Backend rejected the request without a readable error detail."
    text = " ".join(value[:16000].split())
    lower = text.lower()
    if any(
        marker in lower
        for marker in (
            "parse_error",
            "parse error",
            "invalid json",
            "malformed json",
            "failed to parse",
            "json decode",
        )
    ):
        parts = []
        code = re.search(r"json\.exception\.[a-z_]+\.\d+", text, re.I)
        location = re.search(r"line\s+\d+[, :] *column\s+\d+", text, re.I)
        if code:
            parts.append(code.group(0))
        if location:
            parts.append(location.group(0))
        for diagnostic in (
            "unexpected end of input",
            "unterminated string",
            "invalid string",
            "unexpected token",
            "syntax error",
        ):
            if diagnostic in lower:
                parts.append(diagnostic)
                break
        suffix = " (" + "; ".join(parts) + ")" if parts else ""
        return (
            "Malformed JSON in the backend request or tool-call arguments"
            + suffix
            + "."
        )
    # Omit quoted data and everything following an embedded object/list; error
    # prose before those payloads remains available for memory/context failures.
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "[quoted content omitted]", text
    )
    text = re.split(r"[\{\[]", text, maxsplit=1)[0]
    text = re.sub(
        r"(?i)\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{15,})\b",
        "[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)((?:api[_ -]?key|authorization|password|access[_ -]?token)\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    return (
        text.strip()[:360] or "Backend rejected the request; detailed payload omitted."
    )


def _response_error_detail(response) -> str:
    """Prefer structured backend errors, falling back to bounded plain text."""
    try:
        value = response.json()
    except (ValueError, TypeError):
        value = None
    if not isinstance(value, (dict, str)):
        value = response.text
    return _backend_error_detail(value)


class ModelCancelled(ModelError):
    """The user cancelled loading or inference."""


def _cancelled(cancel: Any) -> bool:
    return bool(cancel and (cancel.is_set() if hasattr(cancel, "is_set") else cancel()))


def _check_cancel(cancel: Any) -> None:
    if _cancelled(cancel):
        raise ModelCancelled(
            "Model operation cancelled. Saved task progress is retained."
        )


def _defaults() -> list[dict]:
    """Reuse installed weights and choose conservative single-slot GPU profiles."""
    profiles = []
    for size in ("9B", "4B"):
        folder = MODEL_ROOT / "mradermacher" / f"Huihui-Qwen3.5-{size}-abliterated-GGUF"
        profiles.append(
            {
                "id": f"qwen35-{size.lower()}-abliterated",
                "name": f"Qwen3.5 {size} Abliterated",
                "path": str(folder / f"Huihui-Qwen3.5-{size}-abliterated.Q4_K_M.gguf"),
                "mmproj": str(
                    folder / f"Huihui-Qwen3.5-{size}-abliterated.mmproj-Q8_0.gguf"
                ),
                "context": 32768,
                "gpu_layers": -1,
                "threads": min(6, os.cpu_count() or 2),
                "cache_type": "q8_0",
                "vision": True,
                "thinking": False,
            }
        )
    folder = MODEL_ROOT / "huihui-ai" / "Huihui-Qwen3.8-27B-abliterated-GGUF"
    profiles.append(
        {
            "id": "qwen38-27b-abliterated",
            "name": "Qwen3.8 27B Abliterated",
            "path": str(folder / "Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf"),
            "mmproj": str(folder / "mmproj-model-bf16.gguf"),
            "context": 32768,
            "gpu_layers": 12,
            "threads": min(6, os.cpu_count() or 2),
            "cache_type": "q8_0",
            "vision": False,
            "thinking": False,
        }
    )
    return profiles


class ModelManager:
    """Own a loopback-only inference service without depending on another app."""

    def __init__(self, config_path: Path, state_dir: Path):
        self.config_path = Path(config_path)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log = None
        self._model: dict | None = None
        self._phase = "stopped"
        self._error = ""
        self._device = None
        self.port = int(os.environ.get("LOCAII_MODEL_PORT", "8787"))
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._session = requests.Session()
        self._session.trust_env = (
            False  # Local model prompts must not use HTTP proxies.
        )
        self._api_key = secrets.token_urlsafe(32)
        self._session.headers["Authorization"] = "Bearer " + self._api_key
        if not self.config_path.exists():
            self._save_registry(_defaults())

    def _save_registry(self, profiles: list[dict]) -> None:
        """Atomically persist user-editable profiles without truncating existing data."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "models": profiles}, indent=2) + "\n"
        )
        os.replace(temporary, self.config_path)

    def list_models(self) -> list[dict]:
        with self._lock:
            data = json.loads(self.config_path.read_text())
            profiles = data["models"] if isinstance(data, dict) else data
            return [
                dict(profile, available=Path(profile["path"]).is_file())
                for profile in profiles
            ]

    def register(
        self,
        path: Path,
        name: str,
        context: int = 32768,
        gpu_layers: int = -1,
        mmproj: Path | None = None,
    ) -> dict:
        """Register GGUF weights by reference; never copy or modify the weights."""
        with self._lock:
            path = Path(path).expanduser().resolve(strict=True)
            if path.suffix.lower() != ".gguf" or not path.is_file():
                raise ValueError("Choose a local GGUF model file.")
            if not 2048 <= int(context) <= 262144:
                raise ValueError("Context must be between 2,048 and 262,144 tokens.")
            if not -1 <= int(gpu_layers) <= 999:
                raise ValueError("GPU layers must be -1 (all), or 0 through 999.")
            if mmproj:
                mmproj = Path(mmproj).expanduser().resolve(strict=True)
            profile = {
                "id": hashlib.sha256(str(path).encode()).hexdigest()[:16],
                "name": name.strip() or path.stem,
                "path": str(path),
                "context": int(context),
                "gpu_layers": int(gpu_layers),
                "threads": min(6, os.cpu_count() or 2),
                "mmproj": str(mmproj) if mmproj else None,
                "vision": bool(mmproj),
                "cache_type": "q8_0",
                "thinking": False,
            }
            profiles = [p for p in self.list_models() if p["path"] != str(path)]
            profiles.append(profile)
            self._save_registry(profiles)
            return copy.deepcopy(profile)

    def update(self, model_id: str, **fields) -> dict:
        """Edit a profile atomically; changed settings apply on the next model load."""
        allowed = {
            "name",
            "context",
            "gpu_layers",
            "mmproj",
            "vision",
            "thinking",
            "cache_type",
            "device",
            "threads",
            "idle_timeout",
        }
        if set(fields) - allowed:
            raise ValueError(
                "Unsupported model settings: "
                + ", ".join(sorted(set(fields) - allowed))
            )
        with self._lock:
            profiles = self.list_models()
            profile = next((item for item in profiles if item["id"] == model_id), None)
            if profile is None:
                raise ModelError(f"Unknown model profile: {model_id}")
            if "context" in fields and not 2048 <= int(fields["context"]) <= 262144:
                raise ValueError("Context must be between 2,048 and 262,144 tokens.")
            if "gpu_layers" in fields and not -1 <= int(fields["gpu_layers"]) <= 999:
                raise ValueError("GPU layers must be -1 (all), or 0 through 999.")
            if (
                "idle_timeout" in fields
                and not 30 <= int(fields["idle_timeout"]) <= 7200
            ):
                raise ValueError(
                    "Inference idle timeout must be between 30 and 7,200 seconds."
                )
            if "threads" in fields and not 1 <= int(fields["threads"]) <= 128:
                raise ValueError("CPU threads must be between 1 and 128.")
            if "cache_type" in fields and fields["cache_type"] not in {
                "f16",
                "q8_0",
                "q4_0",
            }:
                raise ValueError("Cache type must be f16, q8_0, or q4_0.")
            if fields.get("mmproj"):
                fields["mmproj"] = str(
                    Path(fields["mmproj"]).expanduser().resolve(strict=True)
                )
            candidate = {**profile, **fields}
            if (
                candidate.get("vision")
                and not Path(candidate.get("mmproj") or "").is_file()
            ):
                raise ValueError("Vision requires a valid local multimodal projector.")
            for key in ("context", "gpu_layers", "threads", "idle_timeout"):
                if key in fields:
                    fields[key] = int(fields[key])
            profile.update(fields)
            self._save_registry(profiles)
            return copy.deepcopy(profile)

    def _runtime(self) -> Path:
        override = os.environ.get("LOCAII_LLAMA_SERVER")
        if override:
            server = Path(override).expanduser()
        else:
            manifest = ROOT / "runtime/runtime.json"
            if not manifest.exists():
                raise ModelError(
                    "Inference runtime is missing. Run scripts/install_runtime.py once."
                )
            server = Path(json.loads(manifest.read_text())["server"])
        if not server.is_file() or not os.access(server, os.X_OK):
            raise ModelError(f"Cannot execute inference runtime: {server}")
        return server

    def _command(self, model: dict) -> list[str]:
        """One slot, modest batches, quantized KV cache, native Jinja tool parsing."""
        server = self._runtime()
        if self._device is None:
            devices = subprocess.run(
                [str(server), "--list-devices"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            for line in devices.stdout.splitlines():
                if "NVIDIA" in line and ":" in line:
                    self._device = line.split(":", 1)[0].strip()
                    break
        command = [
            str(server),
            "--model",
            model["path"],
            "--alias",
            model["id"],
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(model["context"]),
            "--parallel",
            "1",
            "--threads",
            str(model.get("threads", min(6, os.cpu_count() or 2))),
            "--threads-batch",
            str(model.get("threads", min(6, os.cpu_count() or 2))),
            "--n-gpu-layers",
            str(model["gpu_layers"]),
            "--jinja",
            "--batch-size",
            "256",
            "--ubatch-size",
            "128",
            "--flash-attn",
            "on",
            "--cache-type-k",
            model.get("cache_type", "q8_0"),
            "--cache-type-v",
            model.get("cache_type", "q8_0"),
            "--no-webui",
            "--no-agent",
            "--cors-origins",
            "http://127.0.0.1",
        ]
        selected_device = (
            "none"
            if int(model["gpu_layers"]) == 0
            else model.get("device") or self._device
        )
        if selected_device:
            command.extend(["--device", selected_device])
        # Vision projection is opt-in because its buffers materially affect 8 GB GPUs.
        if model.get("vision") and model.get("mmproj"):
            command.extend(
                [
                    "--mmproj",
                    model["mmproj"],
                    "--no-mmproj-offload",
                    "--image-max-tokens",
                    "1024",
                ]
            )
        return command

    def _check_gpu_memory(self, model: dict) -> None:
        """Reject obvious NVIDIA contention before allocating model or KV buffers.

        This is a conservative estimate, not a promise that an arbitrary model
        will fit. Missing NVIDIA utilities never disable Vulkan or CPU support.
        """
        if int(model.get("gpu_layers", -1)) == 0 or model.get("device") == "none":
            return
        selected_device = model.get("device") or self._device
        if not self._device or (
            selected_device != self._device
            and not str(selected_device).startswith("CUDA")
        ):
            return
        executable = shutil.which("nvidia-smi")
        if not executable:
            return
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=index,name,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode:
                return
            devices = list(csv.reader(result.stdout.strip().splitlines()))
            # On multi-GPU systems we cannot infer a Vulkan index from CUDA's
            # enumeration. Let the selected backend make its own allocation.
            if len(devices) != 1 or len(devices[0]) != 4:
                return
            _, name, free, total = [value.strip() for value in devices[0]]
            free_mib, total_mib = float(free), float(total)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return
        weights_mib = Path(model["path"]).stat().st_size / 1024**2
        layers = int(model.get("gpu_layers", -1))
        if layers < 0 or layers >= 999:
            estimate = (
                weights_mib + 768 + 512 * int(model.get("context", 32768)) / 32768
            )
        else:
            # Partial offload varies by architecture. This lower-bound guard
            # catches a busy GPU without claiming to know every tensor layout.
            estimate = min(weights_mib, layers * 192) + 1024
        required = estimate
        if free_mib < required:
            raise ModelError(
                f"{name} has only {free_mib:,.0f} MiB free; this profile needs approximately "
                f"{estimate:,.0f} MiB of GPU headroom. Wait for the other GPU task or unload "
                "its model in that application, then try again. For CPU-only operation set "
                "GPU layers to 0 in the Models page; this explicitly disables device offload."
            )

    def start(self, model_id: str, cancel=None, emit=None) -> dict:
        """Load lazily and wait for health, reporting failure with the backend log tail."""
        with self._lock:
            _check_cancel(cancel)
            profile = next((p for p in self.list_models() if p["id"] == model_id), None)
            if (
                self._model == profile
                and self._process
                and self._process.poll() is None
                and self._phase == "ready"
            ):
                return self.status()
            if not profile:
                raise ModelError(f"Unknown model profile: {model_id}")
            if not Path(profile["path"]).is_file():
                raise ModelError(f'Model weights are missing: {profile["path"]}')
            command = self._command(profile)
            self.stop()
            try:
                self._check_gpu_memory(profile)
            except ModelError as error:
                self._phase, self._error = "error", str(error)
                raise
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    raise ModelError(
                        f"Port {self.port} is already occupied. loCAII will not stop another service."
                    )
            self._phase, self._error = "loading", ""
            self._model = profile
            if emit:
                emit("model_status", {"state": "loading", "model": profile["name"]})
            self._log = (self.state_dir / "llama-server.log").open("ab", buffering=0)
            self._log.write(f'\n--- Loading {profile["name"]} ---\n'.encode())
            env = dict(os.environ)
            # A per-process key prevents arbitrary websites from submitting local
            # inference requests. It is never persisted or printed in command logs.
            env["LLAMA_API_KEY"] = self._api_key
            env.pop("LLAMA_ARG_MCP_SERVERS_CONFIG", None)
            env.pop("LLAMA_ARG_MCP_SERVERS_JSON", None)
            env.pop("LLAMA_ARG_TOOLS", None)
            env["LD_LIBRARY_PATH"] = (
                str(Path(command[0]).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
            )
            try:
                # A separate Python child sets Linux parent-death handling before exec.
                # Avoid preexec_fn because Qt already has other threads running.
                guard = (
                    "import ctypes,os,signal,sys; "
                    "ctypes.CDLL(None).prctl(1,signal.SIGTERM); "
                    "os.getppid()==int(sys.argv[1]) or sys.exit(1); "
                    "os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
                )
                guarded = (
                    [sys.executable, "-c", guard, str(os.getpid()), *command]
                    if sys.platform.startswith("linux")
                    else command
                )
                self._process = subprocess.Popen(
                    guarded,
                    stdout=self._log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                deadline = time.monotonic() + 360
                while time.monotonic() < deadline:
                    _check_cancel(cancel)
                    if self._process.poll() is not None:
                        raise ModelError(
                            "Model failed to load. Reduce GPU layers/context or close another GPU application.\n"
                            + self._log_tail()
                        )
                    try:
                        response = self._session.get(
                            self.base_url + "/health", timeout=2
                        )
                        if response.status_code == 200:
                            self._phase = "ready"
                            if emit:
                                emit("model_status", self.status())
                            return self.status()
                    except requests.RequestException:
                        pass
                    time.sleep(0.3)
                raise ModelError(
                    "Model loading exceeded 6 minutes. See llama-server.log for details."
                )
            except Exception as exc:
                self.stop()
                self._phase, self._error = "error", str(exc)
                raise

    def _log_tail(self) -> str:
        path = self.state_dir / "llama-server.log"
        if not path.exists():
            return ""
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - 5000))
            return source.read().decode(errors="replace")

    def stop(self) -> None:
        """Release only this manager's process, with a bounded shutdown wait."""
        with self._lock:
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            self._process = None
            if self._log:
                self._log.close()
                self._log = None
            self._model = None
            self._phase = "stopped"

    def status(self) -> dict:
        # Do not wait on inference's lock: UI status remains readable during generation.
        process = self._process
        alive = bool(process and process.poll() is None)
        model = self._model
        return {
            "state": self._phase if alive or self._phase == "error" else "stopped",
            "running": alive,
            "model": model["name"] if model else None,
            "model_id": model["id"] if model else None,
            "context": model["context"] if model else 32768,
            "pid": process.pid if alive else None,
            "url": self.base_url,
            "error": self._error,
            "log": str(self.state_dir / "llama-server.log"),
        }

    def chat(
        self, messages: list, tools=None, on_delta=None, cancel=None, max_tokens=2048
    ) -> dict:
        """Collect SSE content and indexed tool fragments into one OpenAI-style message."""
        with self._lock:
            _check_cancel(cancel)
            if not self.status()["running"]:
                raise ModelError("Choose and load a model before sending a message.")
            body = {
                "model": self._model["id"],
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": int(max_tokens),
                "temperature": 0.2,
                "chat_template_kwargs": {
                    "enable_thinking": bool(self._model.get("thinking"))
                },
            }
            if tools:
                body.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
            self._phase = "generating"
            finished = threading.Event()
            interrupt_lock = threading.Lock()
            interrupt_sent = threading.Event()
            owned_process = self._process

            def interrupt_owned():
                # A server draining an aborted prefill can keep HTTP clients
                # alive indefinitely. Its weights are read-only and all task
                # state lives in Python, so discard this owned inference worker.
                with interrupt_lock:
                    if not interrupt_sent.is_set():
                        interrupt_sent.set()
                        self._phase = "stopping"
                        if owned_process and owned_process.poll() is None:
                            owned_process.kill()

            def watch_cancel():
                # Closing an idle HTTP stream can block; terminating our own server
                # also interrupts a stalled prefill immediately and releases VRAM.
                while not finished.wait(0.1):
                    if _cancelled(cancel):
                        interrupt_owned()
                        return

            watcher = threading.Thread(target=watch_cancel, daemon=True)
            watcher.start()
            try:
                with self._session.post(
                    self.base_url + "/v1/chat/completions",
                    json=body,
                    stream=True,
                    timeout=(10, int(self._model.get("idle_timeout", 600))),
                ) as response:
                    if not response.ok:
                        raise ModelError(
                            f"Inference HTTP {response.status_code}: {_response_error_detail(response)}"
                        )
                    content, reasoning, calls, usage, finish = [], [], {}, {}, None
                    for line in response.iter_lines(chunk_size=1):
                        _check_cancel(cancel)
                        if not line or not line.startswith(b"data: "):
                            continue
                        payload = line[6:]
                        if payload == b"[DONE]":
                            break
                        part = json.loads(payload)
                        if part.get("error"):
                            raise ModelError(_backend_error_detail(part["error"]))
                        usage = part.get("usage") or usage
                        for choice in part.get("choices", []):
                            finish = choice.get("finish_reason") or finish
                            delta = choice.get("delta", {})
                            if delta.get("content"):
                                content.append(delta["content"])
                                if on_delta:
                                    on_delta(delta["content"])
                            if delta.get("reasoning_content"):
                                reasoning.append(delta["reasoning_content"])
                            for fragment in delta.get("tool_calls", []):
                                index = fragment.get("index", 0)
                                target = calls.setdefault(
                                    index,
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if fragment.get("id"):
                                    target["id"] = fragment["id"]
                                function = fragment.get("function", {})
                                for key in ("name", "arguments"):
                                    if function.get(key):
                                        target["function"][key] += function[key]
                    _check_cancel(cancel)
                    if finish is None:
                        raise ModelError(
                            "Inference stream ended before a completion marker. Retry the saved task."
                        )
                    result = {"role": "assistant", "content": "".join(content)}
                    if calls:
                        result["tool_calls"] = [calls[index] for index in sorted(calls)]
                    return {
                        "message": result,
                        "usage": usage,
                        "finish_reason": finish,
                        "reasoning": "".join(reasoning),
                    }
            except requests.RequestException as exc:
                _check_cancel(cancel)
                raise ModelError(
                    f"Local inference connection failed: {exc}. Task progress is retained."
                ) from exc
            finally:
                finished.set()
                if _cancelled(cancel):
                    interrupt_owned()
                    if owned_process:
                        owned_process.wait(timeout=5)
                    if self._process is owned_process:
                        self._process = None
                        self._model = None
                        if self._log:
                            self._log.close()
                            self._log = None
                self._phase = "ready" if self.status()["running"] else "stopped"

    def count_tokens(self, messages: list, tools=None) -> int:
        """Measure the actual rendered chat template, including native tool schemas."""
        with self._lock:
            if not self.status()["running"]:
                raise ModelError("Token counting requires a loaded model.")
            body = {
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {
                    "enable_thinking": bool(self._model.get("thinking"))
                },
            }
            if tools:
                body["tools"] = tools
            stage = "/apply-template"
            try:
                rendered = self._session.post(
                    self.base_url + stage, json=body, timeout=30
                )
                rendered.raise_for_status()
                prompt = rendered.json()["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError(
                        "Backend response does not contain a rendered prompt string."
                    )
                stage = "/tokenize"
                tokens = self._session.post(
                    self.base_url + "/tokenize",
                    json={"content": prompt, "add_special": True},
                    timeout=30,
                )
                tokens.raise_for_status()
                token_ids = tokens.json()["tokens"]
                if not isinstance(token_ids, list):
                    raise ValueError(
                        "Backend response does not contain an actual token list."
                    )
                # Image embeddings are not text tokens. Budget above the server's
                # explicit 1,024-token image ceiling, including framing overhead.
                images = sum(
                    1
                    for message in messages
                    if isinstance(message.get("content"), list)
                    for part in message["content"]
                    if part.get("type") == "image_url"
                )
                return len(token_ids) + images * 2048
            except requests.HTTPError as exc:
                response = exc.response
                detail = (
                    _response_error_detail(response)
                    if response is not None
                    else _backend_error_detail(str(exc))
                )
                status = f" HTTP {response.status_code}" if response is not None else ""
                guidance = (
                    " An incomplete tool call in saved history must be repaired before retrying."
                    if stage == "/apply-template" and "Malformed JSON" in detail
                    else ""
                )
                raise ModelError(
                    f"Could not measure the actual model context: {stage}{status}: {detail}"
                    + guidance
                    + " Saved task history is retained; no approximate token count was used."
                ) from None
            except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
                raise ModelError(
                    f"Could not measure the actual model context at {stage}: {_backend_error_detail(str(exc))} "
                    "Saved task history is retained; no approximate token count was used."
                ) from None
