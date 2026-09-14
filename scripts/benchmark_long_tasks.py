#!/usr/bin/env python3
"""Compare long-task survival before and after the 2026-09-14 reliability fixes.

Both versions replay the shape of one recorded real failure: data on a separate
volume, duplicated directory surveys, overlapping file reads, a shell step in the
readable data directory, then a run of path mistakes. Each version runs once in
its own process against an isolated source snapshot, a private temporary store and
a scripted deterministic model. The scripted responses are identical for both
versions, so any difference comes from the engine and tool policy alone.

These are capability and context-cost measurements, not model quality, latency or
statistical performance claims. No model weights, GPU, network, external
supervisor or user workspace files are used.
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

BASELINE = "c705034"
CONTEXT = 16000
MARKER = "29538_session_013_closed_loop"

# Binary capability rows, rendered as a behavior matrix.
CAPABILITIES = (
    ("task_completed", "Task finishes, no abort"),
    ("shell_used_readable_data", "Shell in readable data"),
    ("recovery_guidance", "Corrective guidance sent"),
    ("survey_retained", "Survey survives compaction"),
)
# Quantitative rows; lower is better for every one of them.
COSTS = (
    ("failed_tool_calls", "Failed tool calls"),
    ("redundant_observations", "Duplicate results"),
    ("compactions", "Context archivings"),
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
    """Run the genuine agent loop, tools and durable store with a scripted model."""
    import copy
    import json as js
    import threading

    sys.path.insert(0, str(source))
    # Durable command workers are separate processes; keep them on this snapshot.
    os.environ["PYTHONPATH"] = str(source)
    from locaii.engine import Engine
    from locaii.storage import Store, canonical

    def answer(text="", calls=None, finish="stop"):
        message = {"role": "assistant", "content": text}
        if calls:
            message["tool_calls"] = calls
        return {
            "message": message,
            "finish_reason": finish,
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }

    def call(name, arguments, identifier):
        return {
            "id": identifier,
            "type": "function",
            "function": {"name": name, "arguments": js.dumps(arguments)},
        }

    class ScriptedModel:
        """Fixed responses; exhausting the script yields a final visible answer."""

        def __init__(self, steps):
            self.steps = list(steps)
            self.prompts = []

        def list_models(self):
            return [
                {
                    "id": "synthetic",
                    "name": "Synthetic long-task fixture",
                    "context": CONTEXT,
                    "path": "/synthetic/model.gguf",
                    "gpu_layers": 0,
                }
            ]

        def start(self, identifier, **kwargs):
            return None

        def count_tokens(self, messages, tools=None):
            return len(canonical(messages)) // 4 + 100

        def chat(self, messages, tools=None, **kwargs):
            first = messages[0].get("content", "") if messages else ""
            if first.startswith("Write concise Markdown task memory"):
                return answer("# Memory\nSynthetic evidence retained.")
            self.prompts.append(copy.deepcopy(messages))
            if not self.steps:
                return answer("The dataset is organized by animal and session.")
            return copy.deepcopy(self.steps.pop(0))

    root = output.parent / "state"
    workspace = root / "workspace"
    data = root / "volume" / "LgDel_close_loop_stim1"
    workspace.mkdir(parents=True)
    data.mkdir(parents=True)
    (data / "metadata.csv").write_text(
        "animal_id,session,trial,experiment,condition,arena,date\n"
        + "".join(
            f"29538,baseline,{row},closed_loop,stim,A,2026-09-01\n"
            for row in range(900)
        )
    )
    for index in range(260):
        (data / f"29538_session_{index:03d}_closed_loop").mkdir()
    store = Store(root / "state")
    session = store.create_session("Long task benchmark", workspace)
    identifier = session["id"]

    steps = [
        # The survey, then the same survey again, as observed in the real session.
        answer("", [call("list_directory", {"path": str(data)}, "s1")], "tool_calls"),
        answer("", [call("list_directory", {"path": str(data)}, "s2")], "tool_calls"),
        # Overlapping reads of one metadata file.
        answer(
            "",
            [
                call(
                    "read_text_file",
                    {"path": str(data / "metadata.csv"), "max_chars": 2000},
                    "r1",
                )
            ],
            "tool_calls",
        ),
        answer(
            "",
            [
                call(
                    "read_text_file",
                    {"path": str(data / "metadata.csv"), "max_chars": 500},
                    "r2",
                )
            ],
            "tool_calls",
        ),
        answer(
            "",
            [
                call(
                    "read_text_file",
                    {"path": str(data / "metadata.csv"), "max_chars": 12000},
                    "r3",
                )
            ],
            "tool_calls",
        ),
        answer(
            "",
            [
                call(
                    "read_text_file",
                    {
                        "path": str(data / "metadata.csv"),
                        "offset_chars": 12000,
                        "max_chars": 12000,
                    },
                    "r4",
                )
            ],
            "tool_calls",
        ),
        answer(
            "",
            [
                call(
                    "read_text_file",
                    {
                        "path": str(data / "metadata.csv"),
                        "offset_chars": 24000,
                        "max_chars": 12000,
                    },
                    "r5",
                )
            ],
            "tool_calls",
        ),
        # A shell step inside the directory the task has already read.
        answer(
            "",
            [
                call(
                    "run_command",
                    {"command": "wc -l metadata.csv", "cwd": str(data), "timeout": 20},
                    "c1",
                )
            ],
            "tool_calls",
        ),
        answer(
            "",
            [
                call(
                    "run_command",
                    {
                        "command": "head -1 metadata.csv",
                        "cwd": str(data),
                        "timeout": 20,
                    },
                    "c2",
                )
            ],
            "tool_calls",
        ),
        # Then a run of genuine path mistakes.
        answer("", [call("list_directory", {"path": "./dataset"}, "e1")], "tool_calls"),
        answer(
            "",
            [call("list_directory", {"path": str(workspace / "volume")}, "e2")],
            "tool_calls",
        ),
        answer(
            "", [call("list_directory", {"path": "volume/data"}, "e3")], "tool_calls"
        ),
    ]
    models = ScriptedModel(steps)
    engine = Engine(store, models)
    options = {
        "model_id": "synthetic",
        "max_iterations": 20,
        "workspace_writes": True,
        "allow_terminal": True,
        "read_all_files": True,
        "supervisor_provider": "off",
    }
    status, message = "error", ""
    try:
        result = engine.run(
            identifier,
            "Explain the organization of files inside " + str(data),
            options=options,
        )
        status, message = result.get("status", "unknown"), result.get("text", "")
    except Exception as error:  # Recorded verbatim; this is the measured outcome.
        message = str(error)

    messages = store.messages(identifier)
    receipts = []
    for item in messages:
        if item.get("role") != "tool":
            continue
        try:
            receipts.append(js.loads(item.get("content") or "{}"))
        except ValueError:
            receipts.append({})
    failed = [r for r in receipts if r.get("error") or r.get("success") is False]
    commands = [r for r in receipts if r.get("status") == "completed" and "stdout" in r]
    last_prompt = models.prompts[-1] if models.prompts else []
    final = canonical(last_prompt)
    # Duplicates inside one prompt are the cost the model pays twice. The durable
    # store always keeps every original receipt; only the prompt is condensed.
    prompt_bodies, duplicate_prompt_receipts = set(), 0
    for item in last_prompt:
        if item.get("role") != "tool":
            continue
        body = str(item.get("content") or "")
        if body in prompt_bodies:
            duplicate_prompt_receipts += 1
        prompt_bodies.add(body)
    events = [event["type"] for event in store.events(identifier)]
    # A repeated observation is one whose identical receipt was already recorded.
    seen, redundant, redundant_chars = set(), 0, 0
    for item in messages:
        if item.get("role") != "tool":
            continue
        body = str(item.get("content") or "")
        if body in seen:
            redundant += 1
            redundant_chars += len(body)
        seen.add(body)
    report = {
        "source_digest": source_digest(source),
        "status": status,
        "final_text": message[:200],
        "task_completed": int(status == "completed"),
        "shell_used_readable_data": int(bool(commands)),
        "recovery_guidance": int("recovery" in events),
        "survey_retained": int(MARKER in final),
        "duplicate_survey_copies": final.count(MARKER),
        "redundant_observations": redundant,
        "redundant_receipt_chars": redundant_chars,
        "duplicate_prompt_receipts": duplicate_prompt_receipts,
        "successful_tool_calls": len(receipts) - len(failed),
        "failed_tool_calls": len(failed),
        "compactions": events.count("compaction"),
        "final_prompt_chars": len(final),
        "predictions": len(models.prompts),
        "trial_count": 1,
        "fixture": "synthetic scripted model; real engine, tools and SQLite store",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    store.db.close()


def plot(rows: list[dict], output: Path) -> None:
    """Show handled/not-handled behavior beside the measured context cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "locaii-long-tasks",
        }
    )
    values = [[int(row[key]) for row in rows] for key, _ in CAPABILITIES]
    fig, (matrix, bars) = plt.subplots(
        1, 2, figsize=(10.2, 3.6), facecolor="#f8f9f6", width_ratios=[1.5, 1]
    )
    matrix.imshow(
        values,
        cmap=ListedColormap(["#edf0eb", "#d2e5d7"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    matrix.set_xticks([0, 1], [f"Before ({rows[0]['git_revision'][:7]})", "With fixes"])
    matrix.set_yticks(
        range(len(CAPABILITIES)), [caption for _, caption in CAPABILITIES]
    )
    matrix.tick_params(length=0, pad=8, labelsize=9.5)
    for spine in matrix.spines.values():
        spine.set_visible(False)
    for row in range(len(CAPABILITIES)):
        for column in range(2):
            matrix.text(
                column,
                row,
                "Handled" if values[row][column] else "Not handled",
                ha="center",
                va="center",
                color="#294b3b" if values[row][column] else "#666f68",
                weight="bold",
            )
    positions = range(len(COSTS))
    width = 0.17
    for offset, (row, colour, name) in enumerate(
        [(rows[0], "#b8634f", "Before"), (rows[1], "#3f7d62", "With fixes")]
    ):
        bars.barh(
            [p + (offset - 0.5) * width for p in positions],
            [row[key] for key, _ in COSTS],
            height=width,
            color=colour,
            label=name,
        )
        for index, (key, _) in enumerate(COSTS):
            bars.text(
                row[key] + 0.06,
                index + (offset - 0.5) * width,
                str(row[key]),
                va="center",
                fontsize=9,
                color=colour,
            )
    bars.set_yticks(list(positions), [caption for _, caption in COSTS], fontsize=9.5)
    bars.invert_yaxis()
    bars.set_xlabel("Count (lower is better)", fontsize=9.5)
    bars.legend(
        frameon=False,
        fontsize=9,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.02, 1.0),
        handlelength=1.1,
        columnspacing=1.2,
    )
    bars.tick_params(length=0, labelsize=9)
    bars.margins(y=0.22)
    for side in ("top", "right", "left"):
        bars.spines[side].set_visible(False)
    bars.grid(axis="x", color="#dfe4dd", linewidth=0.7)
    bars.set_axisbelow(True)
    fig.suptitle(
        "Long-task survival: one replayed failure, identical scripted model",
        x=0.02,
        ha="left",
        fontsize=13,
        weight="bold",
        color="#294b3b",
    )
    fig.text(
        0.02,
        0.035,
        "One deterministic trial per version. Real engine, tools and durable store; no GPU, provider or user files.\n"
        "This measures failure handling and context cost, not model quality or speed.",
        fontsize=8.5,
        color="#59655e",
    )
    fig.subplots_adjust(left=0.215, right=0.975, top=0.76, bottom=0.29, wspace=0.60)
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
    """Snapshot both source versions, run isolated probes, write honest artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/benchmarks/long-tasks",
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
    with tempfile.TemporaryDirectory(prefix="locaii-long-task-benchmark-") as temporary:
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
            case = root / version
            case.mkdir(exist_ok=True)
            result_path = case / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--probe-source",
                    str(snapshot / "src"),
                    "--probe-output",
                    str(result_path),
                ],
                cwd=case,
                check=True,
                timeout=180,
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
    if not all(rows[1][key] for key, _ in CAPABILITIES):
        raise SystemExit(
            "Long-task verification failed; inspect the preserved benchmark artifacts."
        )


if __name__ == "__main__":
    main()
