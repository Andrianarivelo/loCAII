#!/usr/bin/env python3
"""Verify actual local image understanding on objectively labelled synthetic shapes.

Defaults to CPU-only 4B. --gpu requires sufficient unoccupied NVIDIA memory and
never stops another application's model. All images and reports are private.
"""

from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid
from PIL import Image, ImageDraw
from locaii.models import ModelManager


def main() -> int:
    """Load matching weights/projector, predict labels, and preserve complete evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--model",
        choices=["qwen35-4b-abliterated", "qwen35-9b-abliterated"],
        default="qwen35-4b-abliterated",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--deadline", type=int, default=240)
    parser.add_argument(
        "--output", type=Path, default=Path(".local/verification/vision")
    )
    args = parser.parse_args()
    os.environ.setdefault("LOCAII_MODEL_PORT", "18787")
    root = args.output.resolve() / uuid.uuid4().hex
    root.mkdir(parents=True)
    picture = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(picture)
    draw.rectangle((48, 76, 148, 176), fill=(230, 20, 30))
    draw.ellipse((352, 76, 452, 176), fill=(20, 45, 230))
    path = root / "vision.png"
    picture.save(path)
    manager = ModelManager(root / "models.json", root / "state")
    model = manager.update(
        args.model,
        context=32768 if args.gpu else 8192,
        gpu_layers=-1 if args.gpu else 0,
        device=None if args.gpu else "none",
        vision=True,
        threads=args.threads,
    )
    cancel = threading.Event()
    timer = threading.Timer(args.deadline, cancel.set)
    started = time.monotonic()

    def gpu_snapshot():
        try:
            return subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return "NVIDIA utility unavailable."

    report = {
        "model": model,
        "mode": "gpu" if args.gpu else "cpu",
        "fixture": str(path),
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ground_truth": {"left": "red square", "right": "blue circle"},
        "gpu_before": gpu_snapshot(),
        "passed": False,
    }
    timer.start()
    try:
        manager.start(
            model["id"],
            cancel=cancel,
            emit=lambda event, payload: print(event, payload, flush=True),
        )
        report["load_seconds"] = time.monotonic() - started
        report["gpu_during"] = gpu_snapshot()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify the color and geometric shape on the LEFT and on the RIGHT of this image. Answer briefly as: Left: color shape. Right: color shape.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(path.read_bytes()).decode()
                        },
                    },
                ],
            }
        ]
        report["budgeted_tokens"] = manager.count_tokens(messages)
        result = manager.chat(messages, cancel=cancel, max_tokens=80)
        report["response"] = result
        clean = re.sub(r"[*_`#]", "", result["message"]["content"].lower())
        report["passed"] = bool(
            re.search(
                r"left\s*:\s*red\s+square.*right\s*:\s*blue\s+circle", clean, re.S
            )
        )
    except Exception as error:
        report["error"] = str(error)
    finally:
        cancel.set()
        timer.cancel()
        manager.stop()
        report["seconds"] = time.monotonic() - started
        report["gpu_after"] = gpu_snapshot()
        target = root / "report.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "report": str(target),
                    "passed": report["passed"],
                    "seconds": report["seconds"],
                }
            ),
            flush=True,
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
