#!/usr/bin/env python3
"""Export measured synthetic workflow checks and archive-compaction figures.

Live reports are read from ignored machine-local verification folders. Only
aggregate timings and pass/fail measurements enter the shareable reports.
A deterministic token proxy measures archive reduction without making claims
about model throughput or the exact tokenizer of a particular GGUF.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
import threading
import time

from locaii.engine import Engine
from locaii.storage import Store, canonical, digest

# All figure styling and evaluation sizes are explicit, reviewable parameters.
CONTEXT_CAPACITY = 32768
RECEIPT_REPEATS = 16000
FIGURE_SIZE = (10.4, 4.0)
COLORS = {"before": "#a7b3ac", "after": "#357f68", "text": "#263a34"}


class TokenProxy:
    """Count UTF-8-independent JSON characters for a deterministic stress test."""

    def count_tokens(self, messages, tools=None):
        return len(canonical([messages, tools or []])) // 4


def compaction_check():
    """Require full archived bytes and constraints to survive a smaller prompt."""
    with tempfile.TemporaryDirectory(prefix="locaii-benchmark-") as temporary:
        root = Path(temporary)
        store = Store(root / "state")
        task = store.create_session(
            "Synthetic archive stress check", root / "workspace"
        )["id"]
        original = "Inspect the synthetic receipt. Preserve all originals. Report only verified observations."
        store.append(task, "user", original)
        store.append(
            task,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "receipt",
                        "type": "function",
                        "function": {
                            "name": "read_text_file",
                            "arguments": '{"path":"synthetic.txt"}',
                        },
                    }
                ],
            },
        )
        store.append(
            task,
            "tool",
            "Synthetic observation. " * RECEIPT_REPEATS + "END_RECEIPT_629",
            tool_call_id="receipt",
        )
        model = TokenProxy()
        engine = Engine(store, model)
        before_messages = store.messages(task)
        before = model.count_tokens(engine._context(task), [])
        started = time.perf_counter()
        reduced, after = engine._compact(
            task,
            engine._context(task),
            [],
            lambda *_: None,
            threading.Event(),
            CONTEXT_CAPACITY,
        )
        elapsed = time.perf_counter() - started
        checkpoint = store.checkpoint(task)
        archived = json.loads(Path(checkpoint["archive"]).read_text())
        checks = {
            "full_archive_matches": digest(archived["messages"])
            == digest(before_messages),
            "source_database_unchanged": store.messages(task) == before_messages,
            "original_constraint_retained": original in canonical(reduced),
            "fits_response_reserve": after < CONTEXT_CAPACITY - 4096,
            "restart_reuses_archive": len(canonical(engine._context(task)))
            < len(canonical(before_messages)),
        }
        store.close()
        return {
            "before_proxy_tokens": before,
            "after_proxy_tokens": after,
            "seconds": round(elapsed, 5),
            "checks": checks,
            "passed": all(checks.values()),
            "method": "Deterministic JSON character count / 4; not model inference.",
        }


def main():
    """Write machine-readable measurements and equivalent PNG/PDF/SVG figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports", type=Path, default=Path(".local/verification/cpu-workflow")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = sorted(
        args.reports.glob("*/report.json"), key=lambda p: p.stat().st_mtime_ns
    )
    for index, path in enumerate(reports, 1):
        report = json.loads(path.read_text())
        checks = report.get("checks", {})
        operations = [
            event
            for event in report.get("events", [])
            if event.get("type") == "tool"
            and event.get("status") == "done"
            and event.get("name") in {"read_text_file", "write_file"}
        ]
        rows.append(
            {
                "run": index,
                "model": report["model"],
                "device": report.get("device", "none"),
                "threads": report.get("threads", 2),
                "context": report.get("context", 8192),
                "run_total_seconds": report.get("seconds"),
                "workflow_seconds": report.get("workflow", {}).get("seconds"),
                "successful_file_operations": len(operations),
                "input_unchanged": report.get("input_unchanged", False),
                "result_verified": checks.get("result_contains_observed_code", False),
                "final_reply_correct": checks.get("visible_final_code", False),
                "workflow_passed": report.get("passed", False),
                "resume_passed": report.get("resume_passed", False),
            }
        )
    if rows:
        with (args.output / "cpu-workflows.csv").open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    resumed = []
    for path in sorted(
        args.reports.glob("*/resume-*.json"), key=lambda p: p.stat().st_mtime_ns
    ):
        report = json.loads(path.read_text())
        resumed.append(
            {
                "model": report["model"],
                "passed": report.get("passed", False),
                "seconds": report.get("seconds"),
                "checks": report.get("checks", {}),
            }
        )
    (args.output / "resume-checks.json").write_text(
        json.dumps(resumed, indent=2) + "\n"
    )
    compact = compaction_check()
    (args.output / "compaction.json").write_text(json.dumps(compact, indent=2) + "\n")
    (args.output / "measurements.json").write_text(
        json.dumps(
            {
                "cpu_workflows": rows,
                "resume_after_lifecycle_fix": resumed,
                "compaction": compact,
                "limitations": "Single runs under concurrent LM Studio CPU/GPU load. No statistical speed claim. Later runs include path guidance and bounded optional memory; model and thread counts may differ.",
            },
            indent=2,
        )
        + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, layout="constrained")
    axes[0].bar(
        ["Before archive", "After archive"],
        [compact["before_proxy_tokens"], compact["after_proxy_tokens"]],
        width=0.42,
        color=[COLORS["before"], COLORS["after"]],
    )
    axes[0].axhline(
        CONTEXT_CAPACITY - 4096,
        color="#bc7960",
        linestyle=":",
        label="Context minus response reserve",
    )
    axes[0].set_title("Compaction retains the full archive", loc="left", weight="bold")
    axes[0].set_ylabel("Token proxy (JSON characters / 4)")
    axes[0].legend(frameon=False, fontsize=8)
    if rows:
        labels = [
            f"Run {row['run']}\n{row['model'].split('-')[1].upper()}, {row['threads']} threads"
            for row in rows
        ]
        bars = axes[1].bar(
            labels,
            [row["successful_file_operations"] for row in rows],
            width=0.42,
            color=[
                COLORS["after"] if row["workflow_passed"] else COLORS["before"]
                for row in rows
            ],
        )
        for bar, row in zip(bars, rows):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                "Strict pass" if row["workflow_passed"] else "Strict fail",
                ha="center",
                fontsize=8,
            )
        axes[1].set_ylim(
            0, max(row["successful_file_operations"] for row in rows) + 1.2
        )
        axes[1].set_ylabel("Observed successful file operations")
    axes[1].set_title("Live Qwen file workflow checks", loc="left", weight="bold")
    figure.supxlabel(
        "Synthetic workspaces only. Single-run checks, not throughput comparisons.",
        fontsize=9,
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(args.output / f"verification.{extension}", dpi=180)
    svg = args.output / "verification.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(figure)
    print(
        json.dumps(
            {
                "reports": len(rows),
                "compaction_passed": compact["passed"],
                "output": str(args.output),
            }
        )
    )
    return 0 if compact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
