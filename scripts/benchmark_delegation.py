#!/usr/bin/env python3
"""Export observed direct/local-supervised checks with explicit comparison limits.

The two single trials use different prompts and memory-generation durations.
They verify functioning workflows, not causal speed improvements or token savings.
Only aggregate measurements and report hashes are published, never private inputs.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

FIGURE_SIZE = (8.4, 3.6)
COLORS = ("#98aa9e", "#426c5b")
OUTPUT = Path("docs/benchmarks")
CAVEAT = (
    "Single trials with different prompts and memory time. No speedup or savings claim."
)


def read(path):
    """Read verification evidence and calculate provenance without publishing paths."""
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main():
    """Compare the prior direct workflow to a real connected supervisor workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--delegated", type=Path, required=True)
    args = parser.parse_args()
    direct, direct_hash = read(args.direct)
    delegated, delegated_hash = read(args.delegated)
    rows = [
        {
            "mode": "Previous direct local workflow",
            "seconds": direct["seconds"],
            "passed": bool(direct["passed"] and direct["resume_passed"]),
            "provider_input_tokens": 0,
            "provider_cached_input_tokens": 0,
            "provider_output_tokens": 0,
            "report_sha256": direct_hash,
        },
        {
            "mode": "Connected supervisor workflow",
            "seconds": delegated["seconds"],
            "passed": delegated["passed"],
            "provider_input_tokens": sum(
                u.get("input_tokens", 0) for u in delegated["provider_usage"]
            ),
            "provider_cached_input_tokens": sum(
                u.get("cached_input_tokens", 0) for u in delegated["provider_usage"]
            ),
            "provider_output_tokens": sum(
                u.get("output_tokens", 0) for u in delegated["provider_usage"]
            ),
            "report_sha256": delegated_hash,
        },
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "delegation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("#f8f9f6")
    ax.set_facecolor("#f8f9f6")
    ax.barh([1, 0], [r["seconds"] for r in rows], height=0.3, color=COLORS)
    ax.set_yticks([1, 0], [r["mode"] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Elapsed seconds, file workflow plus saved-task verification")
    ax.set_xlim(0, max(r["seconds"] for r in rows) * 1.21)
    ax.set_ylim(-0.55, 1.55)
    for y, row in zip([1, 0], rows):
        ax.text(row["seconds"] + 1, y, f"{row['seconds']:.2f} s", va="center")
    fig.suptitle(
        "Observed end-to-end checks · Qwen3.5 9B, 32K, GPU",
        x=0.04,
        ha="left",
        weight="bold",
    )
    fig.text(0.04, 0.06, CAVEAT, fontsize=8.5, color="#617369")
    fig.subplots_adjust(left=0.37, right=0.97, bottom=0.29, top=0.83)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"delegation.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
