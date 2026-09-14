#!/usr/bin/env python3
"""Run the recorded long-task failure shape against a real local model.

A synthetic dataset with the same layout as the user's recorded session is built
outside an empty workspace: animals, sessions, trials and small metadata files.
The task must survey that tree, read a metadata file, use the shell inside it and
report a verification code that exists only in the data.

This check builds its own workspace, dataset and model registry. It never reads
or modifies research files, and never starts, stops or unloads a model owned by
another application. Every dataset file is hashed before and after the run.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

os.environ.setdefault("LOCAII_MODEL_PORT", "18791")
from locaii.engine import Engine
from locaii.models import ModelManager
from locaii.storage import Store

CODE = "LGDEL-4417"
ANIMALS = ["29538", "29539", "29994"]
SESSIONS = ["baseline", "stim_d1", "stim_d7", "maintenance"]


def fingerprint(root: Path) -> dict:
    """Record exact bytes and modification times for every synthetic data file."""
    prints = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            prints[str(path.relative_to(root))] = [
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            ]
    return prints


def build_dataset(root: Path) -> Path:
    """Recreate the observed layout: animal / session / trial / metadata files."""
    for animal in ANIMALS:
        for session in SESSIONS:
            trial = root / animal / session / "1"
            trial.mkdir(parents=True)
            (trial / f"{animal}_1_{session}_metadata.csv").write_text(
                "frame_id,timestamp_camera,animal_id,session,trial,experiment\n"
                + "".join(
                    f"{frame},{frame/30:.3f},{animal},{session},1,closed_loop\n"
                    for frame in range(120)
                )
            )
            (trial / f"{animal}_1_{session}_tracking_dlc.csv").write_text(
                "scorer,time,nose_x,nose_y\nsynthetic,0.0,12.5,44.1\n"
            )
    (root / "README.txt").write_text(
        "Synthetic verification dataset.\n"
        f"Verification code: {CODE}\n"
        "Layout: animal / session / trial / measurement files.\n"
    )
    return root


def main() -> int:
    """Persist a strict machine-readable result even if inference times out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=int, default=600)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--cpu", action="store_true", help="Force a CPU-only profile")
    parser.add_argument(
        "--model",
        choices=["qwen35-4b-abliterated", "qwen35-9b-abliterated"],
        default="qwen35-9b-abliterated",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".local/verification/long-task")
    )
    args = parser.parse_args()

    root = args.output.resolve() / uuid.uuid4().hex
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "notes.md").write_text("Empty analysis workspace.\n")
    dataset = build_dataset(root / "volume" / "LgDel_synthetic_stim1")
    before = fingerprint(dataset)

    store = Store(root / "state")
    manager = ModelManager(root / "models.json", root / "model-runtime")
    manager.update(
        args.model,
        context=args.context,
        gpu_layers=0 if args.cpu else -1,
        device="none" if args.cpu else None,
        vision=False,
        threads=args.threads,
    )
    session = store.create_session("Long task reliability verification", workspace)
    cancel = threading.Event()
    timer = threading.Timer(args.deadline, cancel.set)
    events = []
    started = time.monotonic()
    report = {
        "model": args.model,
        "dataset": str(dataset),
        "workspace": str(workspace),
        "session_id": session["id"],
        "passed": False,
        "checks": {},
        "events": events,
    }

    def emit(kind, payload):
        if kind != "delta":
            event = {
                "seconds": round(time.monotonic() - started, 3),
                "type": kind,
                **payload,
            }
            events.append(event)
            print(json.dumps(event)[:600], flush=True)

    timer.start()
    try:
        result = Engine(store, manager).run(
            session["id"],
            f"Explain the organization of the files inside {dataset}. "
            "Survey the directory tree, inspect one metadata file, and use a shell "
            "command inside that dataset directory to count its animal folders. "
            f"Then reply with SURVEY_READY followed by the verification code written "
            "in the dataset README. Do not modify any dataset file.",
            options={
                "model_id": args.model,
                "workspace_writes": True,
                "read_all_files": True,
                "allow_terminal": True,
                "max_iterations": 30,
                "supervisor_provider": "off",
            },
            cancel=cancel,
            emit=emit,
        )
        report["result"] = result
        tools = [
            event
            for event in events
            if event["type"] == "tool" and event.get("status") in {"done", "error"}
        ]
        commands = [
            event
            for event in events
            if event["type"] == "tool"
            and event.get("name") == "run_command"
            and event.get("status") == "done"
        ]

        def refused(event):
            return "outside the selected workspace" in json.dumps(
                event.get("result", {}), ensure_ascii=False
            )

        # Reads and shell steps must reach readable data. Writes outside the
        # workspace stay refused: that boundary is what protects raw measurements.
        refusals = [
            event
            for event in tools
            if refused(event) and event.get("name") not in {"write_file", "edit_file"}
        ]
        protected = [
            event
            for event in tools
            if refused(event) and event.get("name") in {"write_file", "edit_file"}
        ]
        after = fingerprint(dataset)
        report["checks"] = {
            "completed": result.get("status") == "completed",
            "dataset_unchanged": after == before,
            "shell_used_in_dataset": bool(commands),
            "reads_and_shell_never_refused": not refusals,
            "visible_code": CODE in str(result.get("text", "")),
            "no_unresolved_failures": "unresolved_tool_failures" not in result,
        }
        report["outside_writes_refused"] = len(protected)
        report["tool_calls"] = len(tools)
        report["failed_tool_calls"] = sum(
            1 for event in tools if event.get("status") == "error"
        )
        report["compactions"] = sum(1 for e in events if e["type"] == "compaction")
        report["recoveries"] = sum(1 for e in events if e["type"] == "recovery")
        report["passed"] = all(report["checks"].values())
    except Exception as error:
        report["error"] = str(error)
        report["checks"]["completed"] = False
    finally:
        cancel.set()
        timer.cancel()
        manager.stop()
        report["seconds"] = round(time.monotonic() - started, 3)
        report["messages"] = store.messages(session["id"])
        output = root / "report.json"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "report": str(output),
                    "passed": report["passed"],
                    "checks": report["checks"],
                    "seconds": report["seconds"],
                }
            ),
            flush=True,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
