"""Evaluate real local reasoning on synthetic stale-file and wide-table fixtures."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import uuid

os.environ.setdefault("LOCAII_MODEL_PORT", "18794")
from locaii.engine import Engine
from locaii.models import ModelManager
from locaii.storage import Store

MODEL_ID = "qwen35-9b-abliterated"


def main():
    """Keep all models, files and conversation state isolated from user tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--deadline", type=int, default=180)
    args = parser.parse_args()
    root = Path(".local/verification/evidence-reasoning") / uuid.uuid4().hex
    root.mkdir(parents=True)
    manager = ModelManager(root / "models.json", root / "runtime")
    manager.update(
        MODEL_ID, context=32768, gpu_layers=-1, vision=False, thinking=args.thinking
    )
    source_digest = hashlib.sha256()
    for source_file in sorted(Path("src/locaii").glob("*.py")):
        source_digest.update(source_file.read_bytes())
    source_sha256 = source_digest.hexdigest()
    rows = []
    try:
        for kind in ("header_only", "wide_table"):
            workspace = (root / kind).resolve()
            workspace.mkdir()
            file = workspace / "metadata.csv"
            headers = [f"field_{i}" for i in range(800)] + [
                "mouse_1_center_x",
                "mouse_2_center_x",
            ]
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(headers)
            if kind == "wide_table":
                writer.writerow(["0"] * 800 + ["10", "20"])
                writer.writerow(["0"] * 800 + ["11", "21"])
                file.write_text(buffer.getvalue())
            else:
                file.write_text(buffer.getvalue()[:5000])
            before = hashlib.sha256(file.read_bytes()).hexdigest()
            store = Store(root / f"{kind}-state")
            session = store.create_session(kind, workspace)
            store.append(
                session["id"],
                "user",
                "Inspect whether mouse_1_center_x and mouse_2_center_x contain different coordinate measurements in metadata.csv. Never modify the input.",
            )
            store.append(
                session["id"],
                "assistant",
                "The metadata file is 51 MB. I have normalized its headers. Next I will read the first 2500 characters to inspect coordinate rows.",
            )
            events = []
            cancel = threading.Event()
            timer = threading.Timer(args.deadline, cancel.set)
            timer.start()
            started = time.monotonic()
            try:
                result = Engine(store, manager).run(
                    session["id"],
                    "Continue by checking the available evidence.",
                    options={
                        "model_id": MODEL_ID,
                        "workspace_writes": False,
                        "allow_terminal": False,
                        "read_all_files": False,
                        "supervisor_provider": "off",
                        "max_iterations": 0,
                    },
                    cancel=cancel,
                    emit=lambda kind, payload: (
                        events.append({"type": kind, **payload})
                        if kind != "delta"
                        else None
                    ),
                )
                messages = store.messages(session["id"])
                inspected = [
                    json.loads(m["content"])
                    for m in messages
                    if m.get("role") == "tool" and m.get("name") == "inspect_csv"
                ]
                calls = [
                    call["function"]["name"]
                    for m in messages
                    for call in m.get("tool_calls", [])
                ]
                text = result.get("text", "")
                lower = text.lower()
                factual = (
                    (
                        any(r.get("header_only") for r in inspected)
                        and any(
                            s in lower
                            for s in (
                                "no data",
                                "zero data rows",
                                "no actual data",
                                "only header",
                                "header-only",
                                "no rows",
                                "contains only",
                                "no coordinate",
                            )
                        )
                    )
                    if kind == "header_only"
                    else (
                        any(
                            "mouse_1_center_x" in str(r.get("selected_columns"))
                            and "mouse_2_center_x" in str(r.get("selected_columns"))
                            and r.get("rows")
                            for r in inspected
                        )
                        and any(
                            s in lower for s in ("differ", "distinct", "not identical")
                        )
                    )
                )
                rows.append(
                    {
                        "scenario": kind,
                        "passed": result.get("status")
                        in (
                            {"completed", "needs_attention"}
                            if kind == "header_only"
                            else {"completed"}
                        )
                        and bool(factual)
                        and before == hashlib.sha256(file.read_bytes()).hexdigest(),
                        "seconds": round(time.monotonic() - started, 3),
                        "thinking": args.thinking,
                        "calls": calls,
                        "text": text,
                        "input_unchanged": before
                        == hashlib.sha256(file.read_bytes()).hexdigest(),
                        "result": result,
                        "events": events,
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "scenario": kind,
                        "passed": False,
                        "error": str(error),
                        "seconds": round(time.monotonic() - started, 3),
                        "events": events,
                    }
                )
            finally:
                timer.cancel()
            rows[-1]["source_sha256"] = source_sha256
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in rows[-1].items()
                        if k not in {"events", "result"}
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        manager.stop()
        (root / "report.json").write_text(json.dumps(rows, indent=2))
        print("Report:", root / "report.json", flush=True)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
