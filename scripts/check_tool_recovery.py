#!/usr/bin/env python3
"""Verify native-template recovery without executing a user's research task.

An optional saved session is inspected read-only and passed only to the local
formatter/tokenizer. Real inference runs in a separate synthetic read-only task.
The script never changes the original transcript or executes its pending calls.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid


def main():
    """Load a private 9B worker and verify saved-history formatting and safe resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-db", type=Path)
    parser.add_argument("--saved-session")
    parser.add_argument("--deadline", type=int, default=180)
    args = parser.parse_args()
    os.environ.setdefault("LOCAII_MODEL_PORT", "18793")
    from locaii.engine import Engine, clean_messages
    from locaii.models import ModelManager, ModelError
    from locaii.storage import Store, canonical

    root = Path(".local/verification/tool-recovery").resolve() / uuid.uuid4().hex
    root.mkdir(parents=True)
    manager = ModelManager(root / "models.json", root / "runtime")
    model = "qwen35-9b-abliterated"
    manager.update(
        model, context=32768, gpu_layers=-1, threads=6, vision=False, device=None
    )
    store = Store(root / "state")
    session = store.create_session(
        "Synthetic malformed-call recovery", root / "workspace"
    )
    workspace = Path(session["workspace"])
    code = "RECOVERED_" + uuid.uuid4().hex[:8].upper()
    source = workspace / "input.txt"
    source.write_text(code + "\n")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    broken = {
        "id": "old-broken",
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": '{"path":"never.py","content":"' + ("unfinished " * 1100),
        },
    }
    store.append(
        session["id"],
        "user",
        "Read input.txt and report its code without changing files.",
    )
    store.append(
        session["id"],
        {
            "role": "assistant",
            "content": "An earlier incomplete request.",
            "tool_calls": [broken],
        },
    )
    store.append(
        session["id"],
        "tool",
        '{"error":"Invalid JSON. Nothing was executed."}',
        tool_call_id="old-broken",
    )
    original = store.messages(session["id"])
    cancel = threading.Event()
    timer = threading.Timer(args.deadline, cancel.set)
    started = time.monotonic()
    report = {"passed": False, "checks": {}, "events": []}

    def read_saved():
        db = sqlite3.connect(
            "file:" + str(args.saved_db.resolve()) + "?mode=ro", uri=True
        )
        try:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT payload FROM messages WHERE session_id=? ORDER BY id",
                    (args.saved_session,),
                )
            ]
        finally:
            db.close()

    timer.start()
    try:
        manager.start(model, cancel=cancel)
        if args.saved_db and args.saved_session:
            saved = read_saved()
            fingerprint = hashlib.sha256(canonical(saved).encode()).hexdigest()
            try:
                manager.count_tokens(saved)
                report["original_template_error"] = False
            except ModelError as error:
                report["original_template_error"] = True
                report["bounded_original_diagnostic"] = str(error)[:1000]
            report["repaired_saved_tokens"] = manager.count_tokens(
                clean_messages(saved)
            )
            report["checks"]["actual_saved_history_formats"] = (
                report["repaired_saved_tokens"] > 0
            )
            # Exercise compaction of an isolated copy, without executing any
            # historical task or touching the live session's state/checkpoint.
            from locaii.tools import LocalTools

            copied = store.create_session(
                "Saved-history format-only check", root / "format-only"
            )
            for message in saved:
                store.append(copied["id"], message)
            copied_engine = Engine(store, manager)
            tools = LocalTools(
                store,
                copied["id"],
                {
                    "workspace_writes": False,
                    "allow_terminal": False,
                    "read_all_files": False,
                },
                cancel,
                lambda *args: None,
            )
            compacted, count = copied_engine._compact(
                copied["id"],
                copied_engine._context(copied["id"]),
                tools.schemas(),
                lambda *args: None,
                cancel,
                32768,
            )
            report["compacted_saved_tokens"] = count
            report["checks"]["actual_saved_history_fits_after_compaction"] = (
                count < 32768 - 4096
            )
            report["checks"]["compaction_keeps_raw_copy"] = (
                store.messages(copied["id"]) == saved
            )
            report["checks"]["saved_transcript_unchanged"] = (
                hashlib.sha256(canonical(read_saved()).encode()).hexdigest()
                == fingerprint
            )

        def emit(kind, payload):
            if kind != "delta":
                report["events"].append({"type": kind, **payload})

        result = Engine(store, manager).run(
            session["id"],
            "Continue with the original read-only objective. Use read_text_file on input.txt, then reply with exactly its observed code. Do not write files or regenerate the old script.",
            options={
                "model_id": model,
                "workspace_writes": False,
                "allow_terminal": False,
                "read_all_files": False,
                "supervisor_provider": "off",
                "max_iterations": 6,
            },
            cancel=cancel,
            emit=emit,
        )
        messages = store.messages(session["id"])
        tool_names = [
            m.get("name") for m in messages[len(original) :] if m.get("role") == "tool"
        ]
        report["result"] = result
        report["checks"].update(
            {
                "completed": result["status"] == "completed",
                "observed_code_in_answer": code in result.get("text", ""),
                "real_read_receipt": "read_text_file" in tool_names,
                "no_mutation_tools": not any(
                    n in tool_names
                    for n in ("write_file", "edit_file", "run_command", "start_command")
                ),
                "source_unchanged": hashlib.sha256(source.read_bytes()).hexdigest()
                == source_hash,
                "malformed_script_not_created": not (workspace / "never.py").exists(),
                "original_synthetic_history_unchanged": messages[: len(original)]
                == original,
            }
        )
        report["passed"] = all(report["checks"].values())
    except Exception as error:
        report["error"] = str(error)[:1000]
    finally:
        cancel.set()
        timer.cancel()
        manager.stop()
        report["seconds"] = round(time.monotonic() - started, 3)
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
