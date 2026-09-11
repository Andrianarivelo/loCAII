#!/usr/bin/env python3
"""Export observed GPU checks without publishing private reports or prompts.

Run from the project root after the verification scripts finish. Media reports
are discovered automatically, or selected with repeatable --media-report flags.
Quality is a separate human assessment, never inferred from successful decoding:
  --quality SHA256=pass --quality SHA256=fail
Only approved aggregate fields enter CSV and figures. Failed trials are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

# Figure styling and known, explicitly inspected baseline assessments.
FIGURE_SIZE = (11.8, 5.7)
COLORS = {"cpu": "#a7b3ac", "gpu": "#357f68", "fail": "#a55743", "text": "#263a34"}
QUALITY_FILE = Path("docs/benchmarks/media-quality.json")
FIELDS = (
    "trial",
    "task",
    "model",
    "backend",
    "threads",
    "context",
    "seconds",
    "timing_scope",
    "integrity_status",
    "visual_quality",
    "width",
    "height",
    "frames",
    "steps",
    "fps",
    "decoder",
    "report_sha256",
    "artifact_sha256",
    "notes",
)
LIMITATIONS = (
    "Single trials, no statistical ranking or throughput claim. CPU: 8K / 4 threads; "
    "GPU: 32K / 6 threads. Backend, concurrent load and memory timing differ."
)


def read_report(path):
    """Read a local report and retain a hash for provenance without its path."""
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def result_status(value):
    """Keep absent checks distinct from both a pass and a failure."""
    return "pass" if value is True else "fail" if value is False else "unverified"


def add_row(rows, report_hash, **values):
    """Construct an allowlisted row so report text and local paths cannot leak."""
    row = dict.fromkeys(FIELDS, "")
    row.update(trial=len(rows) + 1, report_sha256=report_hash, **values)
    if isinstance(row["seconds"], (float, int)):
        row["seconds"] = round(row["seconds"], 3)
    rows.append(row)


def workflow_rows(root, mode, rows):
    """Keep every recorded workflow and separate immediate and restarted resume."""
    for path in sorted((root / f"{mode}-workflow").glob("*/report.json")):
        report, fingerprint = read_report(path)
        profile = report.get("profile", {})
        common = {
            "model": report["model"],
            "backend": "CPU" if mode == "cpu" else "Vulkan GPU",
            "threads": report.get("threads", profile.get("threads", "")),
            "context": report.get("context", profile.get("context", "")),
        }
        workflow = report.get("workflow", {})
        add_row(
            rows,
            fingerprint,
            **common,
            task="file_workflow",
            seconds=workflow.get("seconds", report.get("seconds", "")),
            timing_scope=(
                "workflow including optional memory" if workflow else "failed run total"
            ),
            integrity_status=result_status(report.get("passed")),
            notes=(
                "Original input unchanged"
                if report.get("input_unchanged")
                else "Input preservation unverified"
            ),
        )
        if mode == "gpu" and "seconds" in report:
            add_row(
                rows,
                fingerprint,
                **common,
                task="workflow_and_resume_run",
                seconds=report["seconds"],
                timing_scope="complete workflow and immediate resume run",
                integrity_status=result_status(
                    report.get("passed") and report.get("resume_passed")
                ),
            )
        if "resume" in report:
            add_row(
                rows,
                fingerprint,
                **common,
                task="immediate_resume",
                seconds=report["resume"].get("seconds", ""),
                timing_scope="resume including optional memory",
                integrity_status=result_status(report.get("resume_passed")),
            )
        elif report.get("passed") and report.get("resume_passed") is False:
            add_row(
                rows,
                fingerprint,
                **common,
                task="immediate_resume",
                integrity_status="fail",
                timing_scope="resume duration unavailable",
                notes="Original lifecycle failure retained",
            )
        for resumed_path in sorted(path.parent.glob("resume-*.json")):
            resumed, resumed_hash = read_report(resumed_path)
            add_row(
                rows,
                resumed_hash,
                **common,
                task="restart_resume_and_reload",
                seconds=resumed.get("seconds", ""),
                timing_scope="restart resume plus next request",
                integrity_status=result_status(resumed.get("passed")),
                notes="Read-only resume; saved-file integrity and next request checked",
            )


def vision_rows(root, rows):
    """Collect GPU vision as its own task, without comparing different model sizes."""
    for path in sorted((root / "vision").glob("*/report.json")):
        report, fingerprint = read_report(path)
        profile = report.get("model", {})
        if report.get("mode") != "gpu" or not isinstance(profile, dict):
            continue
        add_row(
            rows,
            fingerprint,
            task="vision",
            model=profile.get("id", ""),
            backend="Vulkan GPU; CPU projector",
            threads=profile.get("threads", ""),
            context=profile.get("context", ""),
            seconds=report.get("seconds", ""),
            timing_scope="load plus image response",
            integrity_status=result_status(report.get("passed")),
            notes="Synthetic color and shape grounding; not general visual reasoning",
        )


def media_rows(paths, quality, rows):
    """Separate render integrity from explicitly supplied visual quality verdicts."""
    for path in sorted(set(paths), key=lambda item: item.stat().st_mtime_ns):
        report, fingerprint = read_report(path)
        if not report.get("run_requested"):
            continue
        results = report.get("results", [])
        if not results:
            add_row(
                rows,
                fingerprint,
                task="media_attempt",
                backend="ComfyUI CUDA",
                integrity_status="fail",
                visual_quality="unreviewed",
                notes="Requested generation produced no verified result",
            )
        for result in results:
            receipt = result.get("receipt", {})
            verification = result.get("verification", {})
            params = receipt.get("params", {})
            artifact_hash = verification.get("sha256", "")
            assessment = quality.get(artifact_hash, {})
            verdict = assessment.get("human_inspection", {}).get(
                "verdict", "unreviewed"
            )
            note = {
                "pass": "Inspected output acceptable for this fixture only",
                "fail": "Visual quality failed despite decodable output",
                "unreviewed": "Decoding does not establish visual quality",
            }[verdict]
            add_row(
                rows,
                fingerprint,
                task=result.get("kind", "media"),
                model=receipt.get("workflow", ""),
                backend="ComfyUI CUDA",
                seconds=receipt.get("seconds", ""),
                timing_scope="media job receipt",
                integrity_status=result_status(verification.get("passed")),
                visual_quality=verdict,
                artifact_sha256=artifact_hash,
                decoder=assessment.get("decoder", "unknown"),
                width=verification.get("width", params.get("width", "")),
                height=verification.get("height", params.get("height", "")),
                frames=verification.get("nb_read_frames", params.get("frames", "")),
                steps=params.get("steps", ""),
                fps=params.get("fps", ""),
                notes=note,
            )


def draw_figure(rows, output):
    """Plot matched 9B tasks only, with all media attempts in a quality table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
        }
    )
    media_count = sum(
        row["task"] in {"image", "video", "media_attempt"} for row in rows
    )
    figure = plt.figure(
        figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] + max(0, media_count - 4) * 0.35)
    )
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.65, wspace=0.38)
    for index, (task, title) in enumerate(
        [
            ("file_workflow", "9B: file workflow and memory"),
            ("restart_resume_and_reload", "9B: restarted resume and next request"),
        ]
    ):
        ax = figure.add_subplot(grid[0, index])
        samples = [
            row
            for row in rows
            if row["task"] == task
            and row["model"] == "qwen35-9b-abliterated"
            and isinstance(row["seconds"], (int, float))
        ]
        labels = [
            f"{row['backend']}\n{row['context']//1024}K / {row['threads']} threads"
            for row in samples
        ]
        bars = ax.bar(
            labels,
            [row["seconds"] for row in samples],
            width=0.35,
            color=[
                COLORS["cpu"] if row["backend"] == "CPU" else COLORS["gpu"]
                for row in samples
            ],
        )
        for bar, row in zip(bars, samples):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 4,
                f"{row['seconds']:.2f} s ({row['integrity_status']})",
                ha="center",
                fontsize=8,
            )
        ax.set_ylim(0, max([row["seconds"] for row in samples], default=1) * 1.22)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylabel("Observed wall time (seconds)")
    ax = figure.add_subplot(grid[1, :])
    ax.axis("off")
    media = [row for row in rows if row["task"] in {"image", "video", "media_attempt"}]
    if media:
        cells = [
            [
                f"{row['task'].title()} / {row['steps']} steps / {row['decoder']}",
                f"{row['width']} x {row['height']}",
                str(row["frames"] or "n/a"),
                (
                    f"{row['seconds']:.2f}"
                    if isinstance(row["seconds"], (int, float))
                    else "n/a"
                ),
                row["integrity_status"],
                row["visual_quality"],
            ]
            for row in media
        ]
        table = ax.table(
            cellText=cells,
            colLabels=[
                "Media trial",
                "Dimensions",
                "Frames",
                "Seconds",
                "File integrity",
                "Visual quality",
            ],
            loc="upper center",
            cellLoc="center",
            colWidths=[0.30, 0.16, 0.10, 0.10, 0.17, 0.17],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.6)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#e1e7e3")
            if r == 0:
                cell.set_facecolor("#edf2ee")
            if cell.get_text().get_text() == "fail":
                cell.get_text().set_color(COLORS["fail"])
    ax.set_title(
        "Media trials: file integrity and visual quality are separate checks",
        loc="left",
        weight="bold",
        pad=12,
    )
    figure.subplots_adjust(left=0.075, right=0.98, top=0.9, bottom=0.15)
    figure.text(
        0.075,
        0.055,
        LIMITATIONS
        + "\nMedia timings are receipts, not a comparison of unlike generators. Failed and unreviewed output stays visible.",
        fontsize=8,
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"gpu-verification.{extension}", dpi=180)
    svg = output / "gpu-verification.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(figure)


def main():
    """Export measurements without running inference or touching application state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path(".local/verification"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument(
        "--media-report",
        type=Path,
        action="append",
        help="Repeat to select reports; default discovers all",
    )
    parser.add_argument(
        "--quality",
        action="append",
        default=[],
        metavar="SHA256=VERDICT",
        help="Explicit visual assessment: pass, fail, or unreviewed",
    )
    parser.add_argument(
        "--quality-file",
        type=Path,
        default=QUALITY_FILE,
        help="Explicit visual inspection annotations keyed by artifact SHA256",
    )
    args = parser.parse_args()
    quality = (
        json.loads(args.quality_file.read_text()).get("artifacts", {})
        if args.quality_file.exists()
        else {}
    )
    for assessment in quality.values():
        if assessment.get("human_inspection", {}).get("verdict") not in {
            "pass",
            "fail",
            "unreviewed",
        }:
            parser.error("Quality file contains an invalid inspection verdict")
        if assessment.get("decoder", "unknown") not in {
            "tiled",
            "standard",
            "unknown",
            "n/a",
        }:
            parser.error("Quality file contains an invalid decoder label")
    for specification in args.quality:
        key, separator, verdict = specification.partition("=")
        if (
            len(key) != 64
            or any(c not in "0123456789abcdef" for c in key)
            or not separator
            or verdict not in {"pass", "fail", "unreviewed"}
        ):
            parser.error(
                "--quality requires a lowercase artifact SHA256 and pass, fail, or unreviewed"
            )
        quality.setdefault(key, {})["human_inspection"] = {"verdict": verdict}
    rows = []
    workflow_rows(args.reports, "cpu", rows)
    workflow_rows(args.reports, "gpu", rows)
    vision_rows(args.reports, rows)
    media_rows(
        args.media_report or list((args.reports / "media").glob("*/report.json")),
        quality,
        rows,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "gpu-verification.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    draw_figure(rows, args.output)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "visual_failures": sum(row["visual_quality"] == "fail" for row in rows),
                "limitations": LIMITATIONS,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
