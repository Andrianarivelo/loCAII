#!/usr/bin/env python3
"""Run a bounded real local file workflow, preserving every observed receipt.

CPU is the default. Explicit --gpu validates the NVIDIA profile and refuses
a busy GPU without unloading any model owned by another application.

This check creates its own workspace and model registry. It neither accesses
research files nor starts, stops, or unloads a model in another application.
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

os.environ.setdefault("LOCAII_MODEL_PORT", "18787")
from locaii.engine import Engine
from locaii.models import ModelManager
from locaii.storage import Store


def sha(path: Path) -> str:
    """Measure complete source bytes rather than relying on a model's description."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resume_existing(report_path: Path, deadline: int) -> int:
    """Reopen saved task state and strictly verify read-only continuation after restart."""
    previous = json.loads(report_path.read_text())
    root = report_path.resolve().parent
    workspace = Path(previous["workspace"])
    result_path, source = workspace / "result.md", workspace / "input.txt"
    before = {
        "result_hash": sha(result_path),
        "result_mtime": result_path.stat().st_mtime_ns,
        "input_hash": sha(source),
    }
    store = Store(root / "state")
    manager = ModelManager(root / "models.json", root / "model-runtime")
    identifier, model = previous["session_id"], previous["model"]
    cancel = threading.Event()
    timer = threading.Timer(deadline, cancel.set)
    events = []
    started = time.monotonic()
    report = {
        "source_report": str(report_path.resolve()),
        "model": model,
        "profile": next(
            profile for profile in manager.list_models() if profile["id"] == model
        ),
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
            print(json.dumps(event), flush=True)

    timer.start()
    try:
        result = Engine(store, manager).run(
            identifier,
            "Continue. Read result.md to verify its saved code. Do not rewrite any file. "
            "Then reply RESUME_READY followed by the exact code obtained from that read.",
            options={
                "model_id": model,
                "workspace_writes": True,
                "read_all_files": False,
                "allow_terminal": False,
                "max_iterations": 5,
                "supervisor_provider": "off",
            },
            cancel=cancel,
            emit=emit,
        )
        operations = [
            event["name"]
            for event in events
            if event["type"] == "tool" and event.get("status") == "done"
        ]
        report["resume"] = result
        report["checks"] = {
            "result_unchanged": sha(result_path) == before["result_hash"]
            and result_path.stat().st_mtime_ns == before["result_mtime"],
            "input_unchanged": sha(source) == before["input_hash"],
            "read_observed": "read_text_file" in operations,
            "no_mutation": not any(
                name in {"write_file", "edit_file", "run_command", "start_command"}
                for name in operations
            ),
            "visible_final_code": "RESUME_READY" in result.get("text", "")
            and "BIRCH-629" in result.get("text", ""),
            "completed": result.get("status") == "completed",
        }
        report["status_after_memory"] = manager.status()
        if (
            all(report["checks"].values())
            and time.monotonic() - started < deadline - 25
        ):
            # Exercise immediate reuse after the optional memory timeout. A
            # draining worker must be reaped and a fresh worker loaded here.
            manager.start(model, cancel=cancel, emit=emit)
            followup = manager.chat(
                [{"role": "user", "content": "Reply with exactly RELOAD_READY."}],
                cancel=cancel,
                max_tokens=20,
            )
            report["reload_response"] = followup
            report["checks"]["immediate_next_request"] = (
                "RELOAD_READY" in followup["message"]["content"]
            )
        report["passed"] = all(report["checks"].values()) and report["checks"].get(
            "immediate_next_request", False
        )
    except Exception as error:
        report["error"] = str(error)
    finally:
        cancel.set()
        timer.cancel()
        manager.stop()
        report["seconds"] = round(time.monotonic() - started, 3)
        report["messages"] = store.messages(identifier)
        output = root / ("resume-" + uuid.uuid4().hex + ".json")
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(
            json.dumps(
                {
                    "report": str(output),
                    "passed": report["passed"],
                    "seconds": report["seconds"],
                }
            ),
            flush=True,
        )
    return 0 if report["passed"] else 1


def main() -> int:
    """Persist a strict machine-readable result even if inference times out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=int, default=290)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Explicitly test GPU offload, 32K context, and enabled CPU vision projector",
    )
    parser.add_argument(
        "--context", type=int, help="Override 8K CPU or 32K GPU context"
    )
    parser.add_argument(
        "--model",
        choices=["qwen35-4b-abliterated", "qwen35-9b-abliterated"],
        default="qwen35-4b-abliterated",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".local/verification/cpu-workflow")
    )
    args = parser.parse_args()
    if args.resume_report:
        return resume_existing(args.resume_report, args.deadline)
    root = args.output.resolve() / uuid.uuid4().hex
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    input_path = workspace / "input.txt"
    code = "BIRCH-629"
    input_path.write_text(
        f"This is a synthetic verification file.\nVerification code: {code}\n"
    )
    source_hash = sha(input_path)
    store = Store(root / "state")
    manager = ModelManager(root / "models.json", root / "model-runtime")
    context = args.context or (32768 if args.gpu else 8192)
    manager.update(
        args.model,
        context=context,
        gpu_layers=-1 if args.gpu else 0,
        device=None if args.gpu else "none",
        vision=args.gpu,
        threads=args.threads,
    )
    session = store.create_session(
        ("GPU" if args.gpu else "CPU") + " file workflow verification", workspace
    )
    cancel = threading.Event()
    timer = threading.Timer(args.deadline, cancel.set)
    report = {
        "workspace": str(workspace),
        "session_id": session["id"],
        "model": args.model,
        "device": "NVIDIA auto" if args.gpu else "none",
        "gpu_layers": -1 if args.gpu else 0,
        "mode": "gpu" if args.gpu else "cpu",
        "vision": args.gpu,
        "threads": args.threads,
        "context": context,
        "deadline_seconds": args.deadline,
        "passed": False,
        "resume_passed": False,
        "events": [],
    }
    started = time.monotonic()
    event_path = root / "events.jsonl"

    def emit(kind, payload):
        """Write streamed evidence immediately so partial failures remain inspectable."""
        event = {
            "seconds": round(time.monotonic() - started, 3),
            "type": kind,
            **payload,
        }
        with event_path.open("a") as output:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")
        if kind != "delta":
            print(json.dumps(event), flush=True)
            report["events"].append(event)

    options = {
        "model_id": args.model,
        "workspace_writes": True,
        "read_all_files": False,
        "allow_terminal": False,
        "max_iterations": 8,
        "supervisor_provider": "off",
    }
    timer.start()
    try:
        objective = (
            "Perform exactly this short file task using tools: read input.txt to obtain its verification code; "
            "write result.md containing that exact code; read result.md back to verify it; "
            "then reply WORKFLOW_READY followed by the observed code. Preserve input.txt unchanged. "
            "Use only these three required file operations; no web, terminal, or extra planning is needed."
        )
        result = Engine(store, manager).run(
            session["id"], objective, options=options, cancel=cancel, emit=emit
        )
        report["workflow"] = result
        report["profile"] = next(
            profile for profile in manager.list_models() if profile["id"] == args.model
        )
        transcript = store.messages(session["id"])
        tool_events = [
            event
            for event in report["events"]
            if event["type"] == "tool" and event.get("status") == "done"
        ]
        operations = [event["name"] for event in tool_events]
        final = result.get("text", "")
        result_path = workspace / "result.md"
        checks = {
            "input_unchanged": sha(input_path) == source_hash,
            "result_contains_observed_code": result_path.exists()
            and code in result_path.read_text(),
            "file_tools_observed": "write_file" in operations
            and operations.count("read_text_file") >= 2,
            "visible_final_code": "WORKFLOW_READY" in final and code in final,
            "completed": result.get("status") == "completed",
        }
        report["checks"] = checks
        report["passed"] = all(checks.values())
        report["messages"] = transcript
        if report["passed"] and time.monotonic() - started < args.deadline - 45:
            # Simulate a new engine with the same durable store and exact Continue.
            # A concise added instruction makes the requested verification read-only.
            before_hash, before_mtime = sha(result_path), result_path.stat().st_mtime_ns
            event_index = len(report["events"])
            resumed = Engine(store, manager).run(
                session["id"],
                "Continue. Verify result.md by reading it, do not rewrite any file, then reply RESUME_READY and its code.",
                options=options,
                cancel=cancel,
                emit=emit,
            )
            later = report["events"][event_index:]
            later_operations = [
                event["name"]
                for event in later
                if event["type"] == "tool" and event.get("status") == "done"
            ]
            report["resume"] = resumed
            report["resume_checks"] = {
                "result_unchanged": sha(result_path) == before_hash
                and result_path.stat().st_mtime_ns == before_mtime,
                "input_unchanged": sha(input_path) == source_hash,
                "read_observed": "read_text_file" in later_operations,
                "no_mutation": not any(
                    name in {"write_file", "edit_file", "run_command", "start_command"}
                    for name in later_operations
                ),
                "visible_final_code": "RESUME_READY" in resumed.get("text", "")
                and code in resumed.get("text", ""),
                "completed": resumed.get("status") == "completed",
            }
            report["resume_passed"] = all(report["resume_checks"].values())
        else:
            report["resume_skipped_reason"] = (
                "First workflow failed or less than 45 seconds remained in the bounded run."
            )
    except Exception as error:
        report["error"] = str(error)
    finally:
        cancel.set()
        timer.cancel()
        manager.stop()
        report["seconds"] = round(time.monotonic() - started, 3)
        report["input_unchanged"] = sha(input_path) == source_hash
        report["messages"] = store.messages(session["id"])
        (root / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "report": str(root / "report.json"),
                    "passed": report["passed"],
                    "resume_passed": report["resume_passed"],
                    "seconds": report["seconds"],
                }
            ),
            flush=True,
        )
    return 0 if report["passed"] and report["resume_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
