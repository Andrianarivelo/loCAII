#!/usr/bin/env python3
"""Verify Codex MCP delegation through the real Qt worker and local Qwen model.

This opt-in check uses provider tokens and a free GPU. It creates only synthetic
files in its private workspace, never modifies user datasets or other processes,
and records provider events, local receipts, and independently checked results.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


def main() -> int:
    """Run one bounded supervisor task, preserving evidence even on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the signed-in Codex account and local GPU",
    )
    parser.add_argument("--deadline", type=int, default=360)
    parser.add_argument(
        "--output", type=Path, default=Path(".local/verification/delegation")
    )
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live to run this provider-using integration check.")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("LOCAII_MODEL_PORT", "18789")
    root = args.output.resolve() / uuid.uuid4().hex
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    os.environ["LOCAII_HOME"] = str(root / "state")
    from PySide6.QtWidgets import QApplication
    from locaii.app import build_services
    from locaii.ui import MainWindow, STYLE

    app = QApplication([])
    app.setStyleSheet(STYLE)
    store, engine, models, media = build_services(root / "state")
    models.update(
        "qwen35-9b-abliterated",
        context=32768,
        gpu_layers=-1,
        threads=6,
        device=None,
        vision=True,
    )
    source = workspace / "input.txt"
    code = "MAPLE-" + uuid.uuid4().hex[:6].upper()
    source.write_text("Synthetic fixture. Verification code: " + code + "\n")
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    window = MainWindow(store, engine, models, media)
    window.workspace = str(workspace)
    window.model_combo.setCurrentIndex(
        window.model_combo.findData("qwen35-9b-abliterated")
    )
    window.workspace_writes.setChecked(True)
    window.allow_terminal.setChecked(False)
    window.read_all_files.setChecked(False)
    window.composer.setPlainText("UNSENT USER DRAFT")
    launcher, connection, profile = window._supervisor_connection()
    prepared = launcher.prepare("codex", connection, str(workspace), profile)
    # Use the same session-only config as the interactive launch, with exec for a
    # reproducible bounded test. Keep the user's configured provider/model.
    command = prepared["command"][:-1]
    command.remove("--no-alt-screen")
    command[1:1] = ["exec", "--json", "--ephemeral", "--skip-git-repo-check"]
    objective = (
        "Perform this authorized synthetic integration test now. First inspect local_model_info. "
        "Delegate one simple job to the local model: read input.txt, write result.md with the exact verification code found there, "
        "read result.md back and report its observed code. Preserve input.txt. Poll task_status(wait_seconds=20) until terminal. "
        "Evaluate its actual tool receipts. Then call continue_task for a read-only verification of result.md, asking the local model "
        "to report the code without rewriting files. Poll until terminal and evaluate receipts again. "
        "Use only the loCAII MCP tools. Do not execute this file work yourself. "
        "If both executions completed and receipts support the result, respond DELEGATION_VERIFIED with the observed code; "
        "otherwise explain the actual failure. Do not ask for another objective."
    )
    command.append(objective)
    report = {"passed": False, "workspace": str(workspace), "checks": {}}
    log = (root / "supervisor.jsonl").open("w")
    errors = (root / "supervisor-stderr.txt").open("w")
    start = time.monotonic()
    process = None
    first_result_mtime = None
    try:
        process = subprocess.Popen(
            command,
            cwd=prepared["cwd"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=errors,
            start_new_session=True,
        )
        while time.monotonic() - start < args.deadline:
            app.processEvents()
            state = window._delegation_hub.status()
            jobs = state["jobs"]
            if (
                first_result_mtime is None
                and jobs
                and jobs[0]["status"] == "completed"
                and (workspace / "result.md").exists()
            ):
                first_result_mtime = (workspace / "result.md").stat().st_mtime_ns
            if process.poll() is not None and not window.worker:
                break
            time.sleep(0.04)
        report["provider_exit"] = process.poll()
        report["jobs"] = window._delegation_hub.status()["jobs"]
        output = (root / "supervisor.jsonl").read_text()
        rows = []
        for line in output.splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        report["provider_usage"] = [
            r.get("usage") for r in rows if r.get("type") == "turn.completed"
        ]
        report["provider_messages"] = [
            r.get("item", {}).get("text", "")
            for r in rows
            if r.get("type") == "item.completed"
            and r.get("item", {}).get("type") == "agent_message"
        ]
        jobs = report["jobs"]

        def operations(job):
            return [
                m.get("name")
                for m in store.messages(job["session_id"])
                if m.get("role") == "tool"
            ]

        complete = len(jobs) == 2 and all(j["status"] == "completed" for j in jobs)
        ops = operations(jobs[0]) if jobs else []
        # Session history is shared. Continued messages after the latest user turn
        # identify verification receipts without counting the first execution.
        resumed_tools = []
        if len(jobs) == 2:
            for message in reversed(store.messages(jobs[1]["session_id"])):
                if message.get("role") == "user":
                    break
                if message.get("role") == "tool":
                    resumed_tools.append(message.get("name"))
        report["checks"] = {
            "provider_success": process.poll() == 0,
            "two_completed_local_jobs": complete,
            "same_saved_session": len(jobs) == 2
            and jobs[0]["session_id"] == jobs[1]["session_id"],
            "original_unchanged": hashlib.sha256(source.read_bytes()).hexdigest()
            == original,
            "correct_result_file": (workspace / "result.md").exists()
            and code in (workspace / "result.md").read_text(),
            "actual_write_and_reads": "write_file" in ops
            and ops.count("read_text_file") >= 2,
            "resume_only_read": "read_text_file" in resumed_tools
            and not any(
                n in resumed_tools
                for n in ("write_file", "edit_file", "run_command", "start_command")
            ),
            "resume_preserved_result_mtime": first_result_mtime is not None
            and (workspace / "result.md").stat().st_mtime_ns == first_result_mtime,
            "draft_preserved": window.composer.toPlainText() == "UNSENT USER DRAFT",
            "supervisor_verified_code": any(
                "DELEGATION_VERIFIED" in text and code in text
                for text in report["provider_messages"]
            ),
            "external_review_disabled": all(
                j.get("local_model_tokens", 0) > 0 for j in jobs
            )
            and all(store.supervisor_tokens(j["session_id"]) == 0 for j in jobs),
        }
        report["passed"] = all(report["checks"].values())
    except Exception as error:
        report["error"] = str(error)
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if window.worker:
            window.worker.cancel.set()
            stop = time.monotonic() + 35
            while window.worker and time.monotonic() < stop:
                app.processEvents()
                time.sleep(0.05)
        window.disconnect_supervisor()
        models.stop()
        app.processEvents()
        window.close()
        log.close()
        errors.close()
        report["seconds"] = round(time.monotonic() - start, 3)
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "report": str(root / "report.json"),
                    "passed": report["passed"],
                    "checks": report["checks"],
                    "error": report.get("error"),
                    "seconds": report["seconds"],
                }
            ),
            flush=True,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
