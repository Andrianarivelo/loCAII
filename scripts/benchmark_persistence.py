"""Compare persistent recovery against a prior revision using disposable fixtures.

Run from an IDE or with ``python scripts/benchmark_persistence.py``. The scripted
model deliberately recovers late. This measures engine behavior, not intelligence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading

BASELINE = "f3d4c56"
SCENARIOS = {
    "repeated_reads": (24, "24 unchanged reads"),
    "tool_failures": (12, "12 failed tool calls"),
    "empty_responses": (5, "5 empty responses"),
}
FIGURE_SIZE = (9.0, 3.6)
FIGURE_DPI = 180
COLORS = ("#9DA8B2", "#277568")


def probe(source, destination):
    """Exercise actual engine recovery, substituting only model inference and waits."""
    sys.path.insert(0, str(source))
    from locaii.engine import Engine
    from locaii.storage import Store

    class ScriptedModel:
        """Require more retries than the old stop limits before reading new evidence."""

        def __init__(self, scenario):
            self.scenario = scenario
            self.rounds = 0

        def list_models(self):
            return [{"id": "fixture", "context": 32768}]

        def start(self, *args, **kwargs):
            return None

        def count_tokens(self, *args, **kwargs):
            return 200

        def chat(self, history, tools=None, **kwargs):
            if kwargs.get("cancel") and kwargs["cancel"].is_set():
                raise InterruptedError("Synthetic benchmark deadline reached")
            instruction = str(history[0].get("content", ""))
            if instruction.startswith("Write concise Markdown task memory"):
                return {
                    "message": {
                        "role": "assistant",
                        "content": "Synthetic task checkpoint.",
                    }
                }
            if instruction.startswith("Review task completion using the full sequence"):
                return {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "audit",
                                "type": "function",
                                "function": {
                                    "name": "report_completion",
                                    "arguments": json.dumps(
                                        {
                                            "verdict": "complete",
                                            "summary": "Latest actual tool receipt contains VERIFIED42.",
                                            "next_step": "",
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                }
            self.rounds += 1
            limit = SCENARIOS[self.scenario][0]
            if self.rounds > limit + 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "The verified marker is VERIFIED42.",
                    },
                    "finish_reason": "stop",
                }
            if self.scenario == "empty_responses" and self.rounds <= limit:
                return {"message": {"role": "assistant", "content": ""}}
            arguments = {"path": "fixture.txt", "offset_chars": 256, "max_chars": 256}
            if self.rounds <= limit:
                if self.scenario == "tool_failures":
                    arguments["path"] = f"missing-{self.rounds}.txt"
                else:
                    arguments["offset_chars"] = 0
            return {
                "message": {
                    "role": "assistant",
                    "content": "Inspecting evidence.",
                    "tool_calls": [
                        {
                            "id": str(self.rounds),
                            "type": "function",
                            "function": {
                                "name": "read_text_file",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }

    def skip_wait(self, session_id, cancel, emit, attempt, reason):
        """Remove elapsed backoff only, keeping cancellation effective in the probe."""
        if cancel.is_set():
            raise InterruptedError("Synthetic benchmark deadline reached")

    if hasattr(Engine, "_recovery_wait"):
        Engine._recovery_wait = skip_wait
    rows = []
    for scenario in SCENARIOS:
        with tempfile.TemporaryDirectory(
            prefix="locaii-persistence-probe-"
        ) as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            fixture = workspace / "fixture.txt"
            fixture.write_text("H" * 256 + "VERIFIED42\n")
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            store = Store(root / "state")
            session = store.create_session("Synthetic persistence benchmark", workspace)
            model = ScriptedModel(scenario)
            cancel = threading.Event()
            timer = threading.Timer(20, cancel.set)
            timer.start()
            events = []
            try:
                result = Engine(store, model).run(
                    session["id"],
                    "Find and report the marker on the second page of fixture.txt.",
                    options={
                        "model_id": "fixture",
                        "supervisor_provider": "off",
                        "max_iterations": 0,
                        "persistent_mode": True,
                        "allow_terminal": False,
                        "workspace_writes": False,
                    },
                    cancel=cancel,
                    emit=lambda kind, payload: events.append({"type": kind, **payload}),
                )
                status = result.get("status")
            except Exception as error:
                result = {"error": str(error)}
                status = "interrupted"
            finally:
                timer.cancel()
            messages = store.messages(session["id"])
            verified = any(
                m.get("role") == "tool" and "VERIFIED42" in str(m.get("content"))
                for m in messages
            )
            unchanged = hashlib.sha256(fixture.read_bytes()).hexdigest() == digest
            rows.append(
                {
                    "scenario": scenario,
                    "passed": int(status == "completed" and verified and unchanged),
                    "inference_rounds": model.rounds,
                    "status": status,
                    "verified_evidence": verified,
                    "input_unchanged": unchanged,
                    "result": result,
                    "events": events,
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "rows": rows,
                "source_sha256": {
                    str(path.relative_to(source)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(source.rglob("*.py"))
                },
            },
            indent=2,
        )
        + "\n"
    )


def plot_results(rows, destination):
    """Show three deterministic pass/fail comparisons without statistical claims."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.26, top=0.71, wspace=0.35)
    for axis, (scenario, (_, title)) in zip(axes, SCENARIOS.items()):
        selected = [row for row in rows if row["scenario"] == scenario]
        axis.bar([0, 1], [r["passed"] for r in selected], width=0.3, color=COLORS)
        for index, row in enumerate(selected):
            axis.text(
                index,
                row["passed"] + 0.06,
                "Completed" if row["passed"] else "Interrupted",
                ha="center",
            )
        axis.set(
            title=title,
            ylim=(0, 1.3),
            xticks=[0, 1],
            xticklabels=["Baseline", "Current"],
            yticks=[0, 1],
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E5E9EB", linewidth=0.7)
    axes[0].set_ylabel("Verified completion")
    fig.suptitle(
        "Persistent recovery without manual Continue",
        x=0.06,
        ha="left",
        y=0.97,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.83,
        "Scripted late recovery | Disposable inputs | Backoff waits skipped",
        color="#596973",
    )
    fig.text(
        0.06,
        0.08,
        "Three deterministic engine checks. No GPU inference, sampling, significance tests, or intelligence score.",
        fontsize=8,
        color="#596973",
    )
    for extension in ("png", "pdf", "svg"):
        path = destination.with_suffix("." + extension)
        fig.savefig(path, dpi=FIGURE_DPI, facecolor="white")
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def main():
    """Run baseline and current in separate interpreters and save durable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.probe:
        if not args.evidence:
            parser.error("--probe requires --evidence")
        probe(args.probe, args.evidence)
        return
    repo = Path(__file__).resolve().parents[1]
    private = repo / ".local/verification/persistence"
    private.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(
        prefix="locaii-persistence-baseline-"
    ) as temporary:
        archive = subprocess.check_output(
            ["git", "archive", args.baseline, "src"], cwd=repo
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(temporary, filter="data")
        for label, source in (
            ("baseline", Path(temporary) / "src"),
            ("current", repo / "src"),
        ):
            environment = dict(os.environ, PYTHONPATH=str(source))
            evidence = private / f"{label}.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--probe",
                    str(source),
                    "--evidence",
                    str(evidence),
                ],
                env=environment,
                check=True,
                timeout=80,
            )
            rows.extend(
                {
                    "version": label,
                    **{k: v for k, v in row.items() if k not in {"events", "result"}},
                }
                for row in json.loads(evidence.read_text())["rows"]
            )
    destination = repo / "docs/benchmarks/persistence"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (private / "manifest.json").write_text(
        json.dumps(
            {
                "baseline": subprocess.check_output(
                    ["git", "rev-parse", args.baseline], cwd=repo, text=True
                ).strip(),
                "scope": "Deterministic engine behavior with scripted inference. Source digests are in each probe report.",
            },
            indent=2,
        )
        + "\n"
    )
    plot_results(rows, destination)
    print(json.dumps(rows, indent=2))
    if any(not row["passed"] for row in rows if row["version"] == "current"):
        raise SystemExit("Current implementation failed a persistence regression.")


if __name__ == "__main__":
    main()
