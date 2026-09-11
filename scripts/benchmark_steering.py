#!/usr/bin/env python3
"""Compare actual Qt/Engine steering behavior using an entirely synthetic model.

Each version runs once in its own process against an isolated source snapshot and
private temporary SQLite state. The first fake prediction waits on an event while
the GUI sends an update. The fake model changes its answer only if that exact update
reaches a later prompt. These are deterministic capability checks, not model-quality,
latency, or statistical performance measurements. No model weights, GPU, network,
external supervisor, or user workspace files are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

BASELINE = "a1646cc9bac23663333cc6588ba6022c8bad60d5"
ORIGINAL = "Reply with ORIGINAL_TOKEN. Keep this synthetic task objective."
UPDATE = (
    "Change the final reply to REVISED_TOKEN while retaining the original objective."
)
METRICS = (
    ("accepted_durably", "Update saved during work"),
    ("delivered_to_prompt", "Update reaches next prompt"),
    ("revised_answer_visible", "Revised answer shown"),
)


def source_digest(source: Path) -> str:
    """Fingerprint the exact Python source files exercised by a subprocess."""
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def probe(source: Path, output: Path) -> None:
    """Exercise the genuine GUI, worker, agent loop, and store with a fake predictor."""
    import copy
    import threading
    import time

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, str(source))
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from locaii.engine import Engine
    from locaii.storage import Store, canonical
    from locaii.ui import MainWindow

    class ScriptedModel:
        """A controlled response depends strictly on the prompt actually received."""

        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.prompts = []

        def list_models(self):
            return [
                {
                    "id": "synthetic",
                    "name": "Synthetic steering fixture",
                    "context": 32768,
                    "path": "/synthetic/model.gguf",
                    "gpu_layers": 0,
                }
            ]

        def start(self, *args, **kwargs):
            pass

        def count_tokens(self, messages, tools=None):
            return len(canonical(messages)) // 4 + 100

        def chat(self, messages, tools=None, **kwargs):
            if messages and str(messages[0].get("content", "")).startswith(
                "Write concise Markdown task memory"
            ):
                text = "# Memory\nOriginal synthetic objective and completed reply retained."
            else:
                self.prompts.append(copy.deepcopy(messages))
                if len(self.prompts) == 1:
                    self.entered.set()
                    if not self.release.wait(10):
                        raise RuntimeError("Synthetic release event was not delivered")
                received = any(
                    message.get("role") == "user" and message.get("content") == UPDATE
                    for message in messages
                )
                text = "REVISED_TOKEN" if received else "ORIGINAL_TOKEN"
                if kwargs.get("on_delta"):
                    kwargs["on_delta"](text)
            return {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            }

    class EmptyMedia:
        """Keep media inventory inert so no renderer process is started."""

        def list_workflows(self):
            return []

        def jobs(self):
            return []

    def wait_until(app, condition):
        """Pump Qt events with a diagnostic timeout, never a timing benchmark."""
        deadline = time.monotonic() + 10
        while not condition() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(5)
        if not condition():
            raise RuntimeError(
                "Synthetic GUI operation did not reach the expected state"
            )
        app.processEvents()

    with tempfile.TemporaryDirectory(prefix="locaii-steering-probe-") as temporary:
        root = Path(temporary)
        store = Store(root / "state")
        models = ScriptedModel()
        app = QApplication.instance() or QApplication([])
        window = MainWindow(store, Engine(store, models), models, EmptyMedia())
        window.set_workspace(str(root / "workspace"))
        Path(window.workspace).mkdir(parents=True, exist_ok=True)
        window.composer.setPlainText(ORIGINAL)
        window.send()
        try:
            wait_until(app, models.entered.is_set)
            enabled = window.send_button.isEnabled()
            window.composer.setPlainText(UPDATE)
            window.send()
            pending = (
                store.pending_steering(window.session_id)
                if hasattr(store, "pending_steering")
                else []
            )
            accepted = any(item["text"] == UPDATE for item in pending)
            models.release.set()
            wait_until(app, lambda: window.worker is None)
            # Drain any scheduled completion-race continuation before inspection.
            for _ in range(5):
                app.processEvents()
                QTest.qWait(5)
            wait_until(app, lambda: window.worker is None)
            history = store.messages(window.session_id)
            current_answer = (
                window.stream_message.content if window.stream_message else ""
            )
            delivered = any(
                any(
                    message.get("role") == "user" and message.get("content") == UPDATE
                    for message in prompt
                )
                for prompt in models.prompts[1:]
            )
            report = {
                "source_sha256": source_digest(source),
                "send_enabled_during_work": int(enabled),
                "accepted_durably": int(accepted),
                "delivered_to_prompt": int(delivered),
                "revised_answer_visible": int(current_answer == "REVISED_TOKEN"),
                "original_objective_retained": int(
                    any(
                        message.get("role") == "user"
                        and message.get("content") == ORIGINAL
                        for message in history
                    )
                ),
                "worker_completed": int(
                    store.get_session(window.session_id)["status"] == "completed"
                    and not window.error_seen
                ),
                "task_predictions": len(models.prompts),
                "trial_count": 1,
                "fixture": "synthetic Qt + Engine + SQLite; no inference service",
            }
            output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            models.release.set()
            if window.worker:
                window.stop()
                wait_until(app, lambda: window.worker is None)
            window.close()
            store.db.close()


def plot(rows: list[dict], output: Path) -> None:
    """Render a binary behavior matrix rather than imply quantitative model quality."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "locaii-steering",
        }
    )
    values = [[int(row[metric]) for row in rows] for metric, _ in METRICS]
    fig, ax = plt.subplots(figsize=(7.6, 3.7), facecolor="#f8f9f6")
    ax.imshow(
        values,
        cmap=ListedColormap(["#edf0eb", "#d2e5d7"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks([0, 1], ["Before (a1646cc)", "With live steering"])
    ax.set_yticks(range(len(METRICS)), [caption for _, caption in METRICS])
    ax.tick_params(length=0, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(len(METRICS)):
        for column in range(2):
            ax.text(
                column,
                row,
                "Handled" if values[row][column] else "Not handled",
                ha="center",
                va="center",
                color="#294b3b" if values[row][column] else "#666f68",
                weight="bold",
            )
    fig.suptitle(
        "Live steering: deterministic behavior check",
        x=0.04,
        ha="left",
        fontsize=14,
        weight="bold",
        color="#294b3b",
    )
    fig.text(
        0.04,
        0.035,
        "One synthetic trial per version. Real GUI, worker, agent loop, and durable store.\nNo GPU or provider calls. This does not measure model quality or speed.",
        fontsize=9,
        color="#59655e",
    )
    fig.subplots_adjust(left=0.35, right=0.96, top=0.82, bottom=0.29)
    for extension in ("png", "pdf", "svg"):
        metadata = (
            {"Date": None}
            if extension == "svg"
            else {"CreationDate": None, "ModDate": None} if extension == "pdf" else {}
        )
        fig.savefig(output.with_suffix("." + extension), dpi=180, metadata=metadata)
    svg = output.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def main() -> None:
    """Snapshot both source versions, run isolated probes, and write honest artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/benchmarks/steering",
    )
    parser.add_argument("--probe-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--probe-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_source:
        probe(args.probe_source, args.probe_output)
        return
    project = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline], cwd=project, text=True
    ).strip()
    rows = []
    with tempfile.TemporaryDirectory(prefix="locaii-steering-benchmark-") as temporary:
        root = Path(temporary)
        old = root / "before"
        old.mkdir()
        archive = subprocess.check_output(
            ["git", "archive", revision, "src"], cwd=project
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(old, filter="data")
        current = root / "after"
        shutil.copytree(
            project / "src",
            current / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
        for version, snapshot, git_revision in [
            ("before", old, revision),
            ("after", current, "working-source snapshot"),
        ]:
            result_path = root / (version + ".json")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--probe-source",
                    str(snapshot / "src"),
                    "--probe-output",
                    str(result_path),
                ],
                cwd=root,
                env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                check=True,
                timeout=45,
            )
            rows.append(
                {
                    "version": version,
                    "git_revision": git_revision,
                    **json.loads(result_path.read_text()),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output)
    print(json.dumps(rows, indent=2))
    if not all(rows[1][metric] for metric, _ in METRICS) or not all(
        row["worker_completed"] and row["original_objective_retained"] for row in rows
    ):
        raise SystemExit(
            "Steering verification failed; inspect preserved benchmark artifacts."
        )


if __name__ == "__main__":
    main()
