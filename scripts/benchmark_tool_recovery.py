#!/usr/bin/env python3
"""Compare malformed-tool recovery against commit 247bd8a using strict fake templates.

Two isolated synthetic scenarios run once per source version: a newly generated
truncated write call, and Continue on saved malformed history. Real Engine, Store,
and file tools run in private temporary directories. Only inference is scripted;
template argument parsing is deliberately strict, matching the observed failure.
No GPU, weights, network, external provider, or user's working files are touched.
The artifacts report behavior and source provenance, not speed or model quality.
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from benchmark_steering import source_digest

BASELINE = "247bd8ac86adcba7dd26018d274eba87abcc0277"
RAW = '{"path":"never_execute.py","content":"unfinished'
METRICS = (
    ("completed", "Task completes"),
    ("exact_read_evidence", "Exact file-read evidence"),
    ("raw_history_preserved", "Original raw history preserved"),
    ("invalid_write_not_executed", "Malformed write never executed"),
)


def answer(text="", calls=None, finish="stop"):
    """Construct the same native response envelope for both source versions."""
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return {
        "message": message,
        "finish_reason": finish,
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    }


def tool(name, arguments, identity):
    """Build a native call while allowing the malformed raw fixture explicitly."""
    return {
        "id": identity,
        "type": "function",
        "function": {
            "name": name,
            "arguments": (
                arguments if isinstance(arguments, str) else json.dumps(arguments)
            ),
        },
    }


def probe(source: Path, output: Path) -> None:
    """Run both real agent-loop scenarios against one isolated Python source tree."""
    sys.path.insert(0, str(source))
    from locaii.engine import Engine
    from locaii.storage import Store, canonical

    class StrictTemplateModels:
        """Reject invalid historical arguments before token measurement or generation."""

        def __init__(self, responses):
            self.responses = responses
            self.prompts = []
            self.template_failures = 0

        def list_models(self):
            return [
                {
                    "id": "synthetic",
                    "name": "Strict synthetic template",
                    "context": 32768,
                }
            ]

        def start(self, *args, **kwargs):
            pass

        def validate(self, messages):
            try:
                for message in messages:
                    for call in message.get("tool_calls", []):
                        parsed = json.loads(call["function"]["arguments"])
                        if not isinstance(parsed, dict):
                            raise ValueError("Template arguments must be an object")
                        json.dumps(parsed, allow_nan=False)
            except (ValueError, TypeError, KeyError):
                self.template_failures += 1
                raise

        def count_tokens(self, messages, tools=None):
            self.validate(messages)
            return len(canonical(messages)) // 4 + 100

        def chat(self, messages, tools=None, **kwargs):
            self.validate(messages)
            if messages and str(messages[0].get("content", "")).startswith(
                "Write concise Markdown task memory"
            ):
                return answer(
                    "# Synthetic memory\nOriginal objective and verified receipts retained."
                )
            self.prompts.append(copy.deepcopy(messages))
            if not self.responses:
                raise RuntimeError("Unexpected extra scripted prediction")
            return copy.deepcopy(self.responses.pop(0))

    rows = []
    for scenario in ("live_truncation", "saved_continue"):
        with tempfile.TemporaryDirectory(prefix="locaii-tool-recovery-") as temporary:
            root = Path(temporary)
            store = Store(root / "state")
            session = store.create_session(
                "Synthetic malformed-tool recovery", root / "workspace"
            )
            identity = session["id"]
            workspace = Path(session["workspace"])
            input_path = workspace / "input.txt"
            input_path.write_text("READ_ONLY_RECOVERY")
            original_input = (input_path.read_bytes(), input_path.stat().st_mtime_ns)
            broken = tool("write_file", RAW, "broken")
            original = []
            if scenario == "live_truncation":
                expected = "RECOVERED"
                responses = [
                    answer("Writing a synthetic file", [broken], "length"),
                    answer(
                        calls=[
                            tool(
                                "write_file",
                                {"path": "derived.txt", "content": expected},
                                "fixed",
                            )
                        ],
                        finish="tool_calls",
                    ),
                    answer(
                        calls=[tool("read_text_file", {"path": "derived.txt"}, "read")],
                        finish="tool_calls",
                    ),
                    answer("Verified " + expected),
                ]
                request = "Write and verify a small derived file."
            else:
                expected = "READ_ONLY_RECOVERY"
                store.append(
                    identity, "user", "Read the synthetic input and report the code."
                )
                store.append(
                    identity, answer("Saved malformed attempt", [broken])["message"]
                )
                store.append(
                    identity,
                    "tool",
                    '{"error":"Invalid JSON. Not executed."}',
                    tool_call_id="broken",
                )
                original = copy.deepcopy(store.messages(identity))
                responses = [
                    answer(
                        calls=[tool("read_text_file", {"path": "input.txt"}, "read")],
                        finish="tool_calls",
                    ),
                    answer(expected),
                ]
                request = "Continue. Read input.txt only; do not write."
            models = StrictTemplateModels(responses)
            result = {}
            failure = ""
            try:
                result = Engine(store, models).run(
                    identity,
                    request,
                    options={
                        "model_id": "synthetic",
                        "max_iterations": 10,
                        "workspace_writes": scenario == "live_truncation",
                        "allow_terminal": False,
                        "read_all_files": False,
                        "supervisor_provider": "off",
                    },
                )
            except Exception as exc:
                failure = type(exc).__name__ + ": " + str(exc)
            saved = store.messages(identity)
            raw_preserved = any(
                call.get("id") == "broken"
                and call.get("function", {}).get("arguments") == RAW
                for message in saved
                for call in message.get("tool_calls", [])
            )
            if original:
                raw_preserved = raw_preserved and saved[: len(original)] == original
            receipts = []
            for message in saved:
                if (
                    message.get("role") == "tool"
                    and message.get("tool_call_id") == "read"
                ):
                    try:
                        receipts.append(json.loads(message["content"]))
                    except (ValueError, TypeError):
                        pass
            exact_read = any(receipt.get("content") == expected for receipt in receipts)
            output_path = workspace / (
                "derived.txt" if scenario == "live_truncation" else "input.txt"
            )
            file_matches = output_path.is_file() and output_path.read_text() == expected
            reply = str(result.get("text") or result.get("content") or "")
            rows.append(
                {
                    "scenario": scenario,
                    "source_sha256": source_digest(source),
                    "completed": int(result.get("status") == "completed"),
                    "exact_read_evidence": int(exact_read and file_matches),
                    "recovered_reply": int(expected in reply),
                    "raw_history_preserved": int(raw_preserved),
                    "invalid_write_not_executed": int(
                        not (workspace / "never_execute.py").exists()
                    ),
                    "input_unchanged": int(
                        (input_path.read_bytes(), input_path.stat().st_mtime_ns)
                        == original_input
                    ),
                    "normalized_marker_in_prompt": int(
                        "_locaii_invalid_historical_arguments"
                        in canonical(models.prompts)
                    ),
                    "template_failure_count": models.template_failures,
                    "task_predictions": len(models.prompts),
                    "failure": failure,
                    "trial_count": 1,
                    "fixture": "real Engine, SQLite, file tools; strict scripted template; no inference service",
                }
            )
            store.db.close()
    output.write_text(json.dumps(rows, indent=2) + "\n")


def plot(rows, output):
    """Show binary observed outcomes, separating new calls from resumed history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ordered = [
        next(
            row
            for row in rows
            if row["version"] == version and row["scenario"] == scenario
        )
        for scenario in ("live_truncation", "saved_continue")
        for version in ("before", "after")
    ]
    values = [[row[metric] for row in ordered] for metric, _ in METRICS]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "locaii-tool-recovery",
        }
    )
    fig, ax = plt.subplots(figsize=(8.7, 4.7), facecolor="#f8f9f6")
    ax.imshow(
        values,
        cmap=ListedColormap(["#edf0eb", "#d2e5d7"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(
        range(4),
        [
            "Live call\nbefore",
            "Live call\nfixed",
            "Saved Continue\nbefore",
            "Saved Continue\nfixed",
        ],
    )
    ax.set_yticks(range(len(METRICS)), [caption for _, caption in METRICS])
    ax.tick_params(length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(len(METRICS)):
        for column in range(4):
            ax.text(
                column,
                row,
                "Pass" if values[row][column] else "Fail",
                ha="center",
                va="center",
                weight="bold",
                color="#294b3b" if values[row][column] else "#666f68",
            )
    ax.axvline(1.5, color="#f8f9f6", linewidth=8)
    fig.suptitle(
        "Malformed tool-call recovery",
        x=0.04,
        ha="left",
        fontsize=15,
        weight="bold",
        color="#294b3b",
    )
    fig.text(
        0.04,
        0.065,
        "Before: commit 247bd8a. Fixed: current source snapshot. One synthetic trial per scenario/version.\nStrict fake template, real agent loop and file receipts. No model-quality or speed claim.",
        fontsize=9,
        color="#59655e",
    )
    fig.subplots_adjust(left=0.35, right=0.97, top=0.84, bottom=0.30)
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


def main():
    """Snapshot source privately and retain both failures and successful outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/benchmarks/tool-recovery",
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
    with tempfile.TemporaryDirectory(
        prefix="locaii-tool-recovery-benchmark-"
    ) as temporary:
        root = Path(temporary)
        before = root / "before"
        before.mkdir()
        archive = subprocess.check_output(
            ["git", "archive", revision, "src"], cwd=project
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(before, filter="data")
        after = root / "after"
        shutil.copytree(
            project / "src",
            after / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
        for version, snapshot in (("before", before), ("after", after)):
            result = root / (version + ".json")
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--probe-source",
                    str(snapshot / "src"),
                    "--probe-output",
                    str(result),
                ],
                cwd=root,
                env=os.environ.copy(),
                check=True,
                timeout=45,
            )
            rows.extend(
                {
                    "version": version,
                    "git_revision": (
                        revision if version == "before" else "working-source snapshot"
                    ),
                    **row,
                }
                for row in json.loads(result.read_text())
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output)
    print(json.dumps(rows, indent=2))
    required = [name for name, _ in METRICS] + ["recovered_reply", "input_unchanged"]
    if not all(
        row[name] for row in rows if row["version"] == "after" for name in required
    ):
        raise SystemExit(
            "Current recovery verification failed; inspect preserved artifacts."
        )


if __name__ == "__main__":
    main()
