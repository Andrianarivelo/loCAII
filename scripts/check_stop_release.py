"""Verify Stop against an owned GPU model using a private synthetic request."""

from pathlib import Path
import json
import os
import subprocess
import threading
import time
import uuid

os.environ.setdefault("LOCAII_MODEL_PORT", "18793")
from locaii.models import ModelManager

OUTPUT = Path(".local/verification/stop-release")
MODEL_ID = "qwen35-9b-abliterated"
DEADLINE_SECONDS = 90


def gpu_memory():
    """Read total GPU usage without affecting other applications."""
    return int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()[0]
    )


def main():
    """Load, cancel active inference, and verify child exit and VRAM release."""
    root = OUTPUT / uuid.uuid4().hex
    root.mkdir(parents=True)
    manager = ModelManager(root / "models.json", root / "runtime")
    manager.update(MODEL_ID, context=32768, gpu_layers=-1, vision=False)
    cancel = threading.Event()
    outcome = []
    worker = None
    report = {"passed": False, "gpu_before_mib": gpu_memory()}

    def inference():
        """Keep generation active long enough to exercise cancellation."""
        try:
            manager.chat(
                [
                    {
                        "role": "user",
                        "content": "Write a long numbered list of integers, one per line, starting at 1. Continue to 10000.",
                    }
                ],
                cancel=cancel,
                max_tokens=6000,
            )
            outcome.append("completed_before_stop")
        except Exception as error:
            outcome.append(type(error).__name__)

    timer = threading.Timer(DEADLINE_SECONDS, cancel.set)
    timer.start()
    try:
        manager.start(MODEL_ID, cancel=cancel)
        process = manager._process
        report["owned_pid"] = process.pid
        report["gpu_loaded_mib"] = gpu_memory()
        worker = threading.Thread(target=inference, daemon=True)
        worker.start()
        time.sleep(1)
        started = time.monotonic()
        cancel.set()
        manager.stop()
        worker.join(10)
        report["stop_seconds"] = round(time.monotonic() - started, 3)
        report["owned_process_exited"] = process.poll() is not None
        report["inference_thread_exited"] = not worker.is_alive()
        report["outcome"] = outcome
        # Driver accounting can lag process teardown briefly.
        for _ in range(30):
            report["gpu_after_mib"] = gpu_memory()
            if report["gpu_after_mib"] <= report["gpu_before_mib"] + 100:
                break
            time.sleep(0.1)
        report["passed"] = (
            report["owned_process_exited"]
            and report["inference_thread_exited"]
            and report["gpu_after_mib"] <= report["gpu_before_mib"] + 100
            and outcome == ["ModelCancelled"]
            and report["stop_seconds"] < 10
        )
    finally:
        cancel.set()
        manager.stop()
        timer.cancel()
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("Report:", root / "report.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
