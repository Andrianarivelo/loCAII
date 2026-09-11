#!/usr/bin/env python3
"""Verify a mid-task GUI update with a real local model, using CPU by default.

Only a private synthetic chat is created. No external supervisor is enabled and
no foreign process is stopped. The update is sent through the real UI send action
when the worker announces its first prediction, rather than before task launch.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import uuid


def main():
    """Run the GUI worker, inject a unique correction, and verify durable delivery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen35-4b-abliterated")
    parser.add_argument("--deadline", type=int, default=220)
    parser.add_argument(
        "--output", type=Path, default=Path(".local/verification/steering")
    )
    args = parser.parse_args()
    root = args.output.resolve() / uuid.uuid4().hex
    root.mkdir(parents=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["LOCAII_HOME"] = str(root / "state")
    os.environ.setdefault("LOCAII_MODEL_PORT", "18791")
    from PySide6.QtWidgets import QApplication
    from locaii.app import build_services
    from locaii.ui import MainWindow

    app = QApplication([])
    store, engine, models, media = build_services(root / "state")
    models.update(
        args.model, context=8192, gpu_layers=0, device="none", vision=False, threads=4
    )
    window = MainWindow(store, engine, models, media)
    window.set_workspace(str(root / "workspace"))
    window.model_combo.setCurrentIndex(window.model_combo.findData(args.model))
    window.supervisor_provider.setCurrentIndex(
        window.supervisor_provider.findData("off")
    )
    window.allow_terminal.setChecked(False)
    window.read_all_files.setChecked(False)
    code = "STEERED_" + uuid.uuid4().hex[:8].upper()
    update = "Change the response: reply with exactly " + code + ". Do not use tools."
    original_handler = window.handle_event
    events = []
    sent = False
    started = time.monotonic()
    report = {"passed": False, "model": args.model, "backend": "CPU", "checks": {}}

    def handler(event, payload):
        """Submit actual user steering from Qt's main thread during prediction."""
        nonlocal sent
        original_handler(event, payload)
        if event != "delta":
            events.append({"event": event, "payload": payload})
        if (
            event == "status"
            and str(payload.get("text", "")).startswith("Working · step")
            and not sent
        ):
            sent = True
            report["sent_while_worker_active"] = window.worker is not None
            report["send_update_enabled"] = window.send_button.isEnabled()
            window.composer.setPlainText(update)
            window.send()

    window.handle_event = handler
    try:
        window.composer.setPlainText(
            "Describe a mountain in two short sentences. Do not use tools."
        )
        window.send()
        while time.monotonic() - started < args.deadline:
            app.processEvents()
            if not window.worker:
                app.processEvents()
                if not window.worker:
                    break
            time.sleep(0.025)
        session = window.session_id
        messages = store.messages(session) if session else []
        last = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "assistant"
            ),
            "",
        )
        delivered = [m for m in messages if m.get("_steering_id")]
        report["checks"] = {
            "sent_during_worker": sent
            and report.get("sent_while_worker_active", False),
            "send_enabled": report.get("send_update_enabled", False),
            "delivered_exactly_once": len(delivered) == 1
            and delivered[0]["content"] == update,
            "no_pending_messages": bool(session)
            and not store.pending_steering(session),
            "completed": bool(session)
            and store.get_session(session)["status"] == "completed",
            "revised_final_answer": code in last,
            "visible_applied_event": any(
                e["event"] == "steering_applied" for e in events
            ),
            "no_external_reviews": bool(session)
            and store.supervisor_tokens(session) == 0,
        }
        report["passed"] = all(report["checks"].values())
        report["messages"] = messages
        report["events"] = events
        report["session_id"] = session
    except Exception as error:
        report["error"] = str(error)
    finally:
        if window.worker:
            window.stop()
            deadline = time.monotonic() + 35
            while window.worker and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.05)
        models.stop()
        window.close()
        app.processEvents()
        report["seconds"] = round(time.monotonic() - started, 3)
        (root / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
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
