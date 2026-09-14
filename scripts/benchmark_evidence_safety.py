"""Compare deterministic evidence and data-protection mechanisms on synthetic files.

Launch from an IDE or run ``python scripts/benchmark_evidence_safety.py``.
No model, GPU, real dataset, or live task is used. Each check is one reproducible
regression scenario, not an estimate of model intelligence or statistical accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading

# All benchmark and figure parameters are grouped here for straightforward review.
BASELINE = "5571c2b"
OUTPUT_STEM = "evidence-safety"
PRIVATE_DIRECTORY = ".local/verification/evidence-safety"
FIGURE_SIZE = (8.8, 3.5)
FIGURE_DPI = 180
COLORS = ("#9DA8B2", "#277568")
SCENARIOS = {
    "latest_receipt": "Keep newest evidence",
    "source_integrity": "Preserve original bytes",
    "header_only": "Detect header-only CSV",
}


def sha256_bytes(value):
    """Return a reproducible content digest for files and synthetic inputs."""
    return hashlib.sha256(value).hexdigest()


def source_digests(source):
    """Identify the exact Python sources exercised, including uncommitted edits."""
    return {
        str(path.relative_to(source)): sha256_bytes(path.read_bytes())
        for path in sorted(source.rglob("*.py"))
    }


def probe(source, destination):
    """Run all three actual implementation checks in an isolated interpreter."""
    sys.path.insert(0, str(source))
    from locaii.engine import collapse_receipts
    from locaii.storage import Store
    from locaii.tools import LocalTools

    with tempfile.TemporaryDirectory(prefix="locaii-evidence-probe-") as folder:
        root = Path(folder)
        workspace = root / "workspace"
        workspace.mkdir()
        store = Store(root / "private-state")
        session = store.create_session("Synthetic evidence benchmark", workspace)
        local = LocalTools(
            store,
            session["id"],
            {"workspace_writes": True, "allow_terminal": True, "read_all_files": True},
            threading.Event(),
            lambda *_: None,
        )

        # Fixture preparation deliberately changes a synthetic file between reads.
        # The second receipt must remain authoritative even though it is shorter.
        changing = workspace / "changing_metadata.csv"
        changing.write_text("coordinate_a,coordinate_b\n" + "1,2\n" * 5000)
        first = local.execute(
            "read_text_file", {"path": str(changing), "max_chars": 6000}
        )
        changing.write_text("H" * 5000)
        latest = local.execute(
            "read_text_file", {"path": str(changing), "max_chars": 5000}
        )
        history = []
        for identifier, result in (("older", first), ("latest", latest)):
            history.extend(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": identifier,
                                "type": "function",
                                "function": {
                                    "name": "read_text_file",
                                    "arguments": json.dumps(
                                        {"path": str(changing), "offset_chars": 0}
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": identifier,
                        "content": json.dumps(result),
                    },
                ]
            )
        collapsed = collapse_receipts(history)
        retained = json.loads(collapsed[-1]["content"])
        newest_kept = retained.get("size_bytes") == 5000 and not retained.get(
            "superseded"
        )

        # Reproduce read-prefix-then-rewrite only on a disposable synthetic source.
        # PYTHONPATH is inherited by LocalTools' independent durable job worker.
        source_file = root / "synthetic_original_metadata.csv"
        original = b"mouse_1_center_x,mouse_2_center_x\n" + b"1,2\n" * 4000
        source_file.write_bytes(original)
        script = (
            "from pathlib import Path; "
            f"p = Path({str(source_file)!r}); "
            "prefix = p.read_text()[:5000]; p.write_text(prefix)"
        )
        receipt = local.execute(
            "run_command",
            {
                "command": f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
                "cwd": str(workspace),
                "timeout": 10,
            },
        )
        after = source_file.read_bytes()
        source_preserved = after == original
        # Count this as protection only when a real attempted write was refused.
        protected = source_preserved and receipt.get("status") == "failed"

        header = workspace / "header_only.csv"
        header.write_text("mouse_1_center_x,mouse_2_center_x\n")
        inspection = local.execute("inspect_csv", {"path": str(header)})
        header_detected = (
            inspection.get("header_only") is True
            and inspection.get("total_data_rows") == 0
        )
        rows = [
            {
                "scenario": "latest_receipt",
                "passed": int(newest_kept),
                "observed": (
                    "latest receipt retained"
                    if newest_kept
                    else "newer shorter receipt hidden"
                ),
            },
            {
                "scenario": "source_integrity",
                "passed": int(protected),
                "observed": f"{len(original)} to {len(after)} bytes; command {receipt.get('status')}",
            },
            {
                "scenario": "header_only",
                "passed": int(header_detected),
                "observed": (
                    "zero data rows identified"
                    if header_detected
                    else "native table inspection unavailable or inconclusive"
                ),
            },
        ]
        evidence = {
            "scope": "Deterministic mechanism regressions, not model-quality measurement.",
            "source_digests_sha256": source_digests(source),
            "rows": rows,
            "receipt_check": {
                "first_size": first.get("size_bytes"),
                "latest_size": latest.get("size_bytes"),
                "retained_latest": retained,
            },
            "source_check": {
                "before_sha256": sha256_bytes(original),
                "after_sha256": sha256_bytes(after),
                "command_receipt": receipt,
            },
            "header_only_receipt": inspection,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(evidence, indent=2) + "\n")


def plot_results(rows, destination, baseline):
    """Draw compact binary checks without implying inferential statistics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"}
    )
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE)
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.25, top=0.71, wspace=0.35)
    for axis, (scenario, label) in zip(axes, SCENARIOS.items()):
        selected = [row for row in rows if row["scenario"] == scenario]
        values = [row["passed"] for row in selected]
        axis.bar([0, 1], values, color=COLORS, width=0.30)
        for index, value in enumerate(values):
            axis.text(
                index,
                value + 0.065,
                "Pass" if value else "Fail",
                ha="center",
                color="#263740",
                fontsize=10,
            )
        axis.set(
            title=label,
            ylim=(0, 1.30),
            xticks=[0, 1],
            xticklabels=["Baseline", "Current"],
            yticks=[0, 1],
            yticklabels=["0", "1"],
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#CDD3D6")
        axis.tick_params(length=0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E5E9EB", linewidth=0.7)
    axes[0].set_ylabel("Regression check passed")
    figure.suptitle(
        "Evidence handling and source-file protection",
        x=0.06,
        ha="left",
        y=0.98,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.06,
        0.84,
        f"Baseline {baseline} vs current source | Disposable synthetic files only",
        color="#596973",
    )
    figure.text(
        0.06,
        0.08,
        "Three deterministic mechanism checks. No model inference, sampling, significance tests, or intelligence score.",
        fontsize=8,
        color="#596973",
    )
    for extension in ("png", "pdf", "svg"):
        path = destination.with_suffix("." + extension)
        figure.savefig(path, dpi=FIGURE_DPI, facecolor="white")
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(figure)


def main():
    """Archive the baseline, execute both sources, and save inspectable results."""
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
    private = repo / PRIVATE_DIRECTORY
    private.mkdir(parents=True, exist_ok=True)
    baseline_commit = subprocess.check_output(
        ["git", "rev-parse", args.baseline], cwd=repo, text=True
    ).strip()
    rows = []
    with tempfile.TemporaryDirectory(prefix="locaii-evidence-baseline-") as temporary:
        archive = subprocess.check_output(
            ["git", "archive", args.baseline, "src"], cwd=repo
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(temporary, filter="data")
        for label, source in (
            ("baseline", Path(temporary) / "src"),
            ("current", repo / "src"),
        ):
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(source)
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
                timeout=45,
            )
            measured = json.loads(evidence.read_text())
            rows.extend({"version": label, **row} for row in measured["rows"])
    destination = repo / "docs" / "benchmarks" / OUTPUT_STEM
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["version", "scenario", "passed", "observed"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    (private / "manifest.json").write_text(
        json.dumps(
            {
                "baseline_commit": baseline_commit,
                "current_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                "scope": "Mechanism regressions only. Per-source digests are in baseline.json and current.json.",
            },
            indent=2,
        )
        + "\n"
    )
    plot_results(rows, destination, args.baseline)
    print(json.dumps(rows, indent=2))
    if any(not row["passed"] for row in rows if row["version"] == "current"):
        raise SystemExit("Current source failed an evidence-safety regression check.")


if __name__ == "__main__":
    main()
