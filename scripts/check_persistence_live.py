"""Verify an uninterrupted real-model task on disposable inputs and exact outputs.

Run ``python scripts/check_persistence_live.py`` from the repository environment.
The deadline limits this verification run, not ordinary app tasks. Private state
and an owned model process keep production conversations and source files intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid

MODEL_ID = "qwen35-9b-abliterated"
MODEL_PORT = 18794
DEADLINE_SECONDS = 300
CONTEXT_TOKENS = 32768
EXPECTED_OUTPUT = {"alpha": 17, "beta": 25, "total": 42}


def digest(path):
    """Hash an exact fixture without changing its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Read two files, persist the computed result, and verify the model's finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=int, default=DEADLINE_SECONDS)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    # Refuse an occupied port rather than disturbing someone else's model server.
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", MODEL_PORT)) == 0:
            raise SystemExit(f"Verification port {MODEL_PORT} is already in use.")
    os.environ["LOCAII_MODEL_PORT"] = str(MODEL_PORT)
    from locaii.engine import Engine
    from locaii.models import ModelManager
    from locaii.storage import Store

    repo = Path(__file__).resolve().parents[1]
    root = repo / ".local/verification/persistence-live" / uuid.uuid4().hex
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    inputs = {
        workspace / "alpha.txt": "count=17\n",
        workspace / "beta.txt": "count=25\n",
    }
    for path, content in inputs.items():
        path.write_text(content)
    before = {path.name: digest(path) for path in inputs}
    store = Store(root / "state")
    session = store.create_session("Synthetic autonomous completion", workspace)
    manager = ModelManager(root / "models.json", root / "runtime")
    manager.update(
        MODEL_ID,
        context=CONTEXT_TOKENS,
        gpu_layers=-1,
        vision=False,
        thinking=args.thinking,
    )
    events = []
    cancel = threading.Event()
    timer = threading.Timer(args.deadline, cancel.set)
    result = {}
    report = {
        "passed": False,
        "model_id": MODEL_ID,
        "thinking": args.thinking,
        "deadline_seconds": args.deadline,
        "source_sha256": {
            str(p.relative_to(repo)): digest(p)
            for p in sorted((repo / "src/locaii").glob("*.py"))
        },
    }
    started = time.monotonic()
    timer.start()
    try:
        result = Engine(store, manager).run(
            session["id"],
            "Complete this task autonomously without asking me to Continue. Read alpha.txt and beta.txt. "
            "Each contains count=N. Create result.json with keys alpha and beta containing integer counts, "
            "and total containing their sum. Read the saved result.json back and verify all three values against "
            "the input files before your final answer. Keep both inputs unchanged. Finish only once all steps are done.",
            options={
                "model_id": MODEL_ID,
                "persistent_mode": True,
                "max_iterations": 0,
                "workspace_writes": True,
                "allow_terminal": False,
                "read_all_files": False,
                "supervisor_provider": "off",
            },
            cancel=cancel,
            emit=lambda kind, payload: (
                events.append({"type": kind, **payload}) if kind != "delta" else None
            ),
        )
        messages = store.messages(session["id"])
        receipts = []
        for message in messages:
            if message.get("role") == "tool":
                try:
                    receipts.append(
                        {
                            "name": message.get("name"),
                            "result": json.loads(message["content"]),
                        }
                    )
                except (TypeError, ValueError):
                    pass
        output = workspace / "result.json"
        saved = json.loads(output.read_text()) if output.exists() else None
        read_back = any(
            r["name"] == "read_text_file"
            and Path(r["result"].get("path", "")).name == "result.json"
            and "42" in str(r["result"].get("text", r["result"].get("content", "")))
            for r in receipts
        )
        report.update(
            result=result,
            saved_output=saved,
            read_back_verified=read_back,
            input_unchanged=before == {path.name: digest(path) for path in inputs},
            receipts=receipts,
        )
        report["passed"] = bool(
            result.get("status") == "completed"
            and saved == EXPECTED_OUTPUT
            and read_back
            and report["input_unchanged"]
        )
    except Exception as error:
        report["error"] = str(error)
    finally:
        timer.cancel()
        report["seconds"] = round(time.monotonic() - started, 3)
        report["deadline_reached"] = cancel.is_set()
        manager.stop()
        report["model_released"] = not manager.status().get("running")
        report["events"] = events
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"events", "receipts", "source_sha256"}
            },
            indent=2,
        )
    )
    print("Report:", root / "report.json", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
