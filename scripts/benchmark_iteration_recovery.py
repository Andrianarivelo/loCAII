"""Compare scripted loop behavior against a Git revision using synthetic data."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

BASELINE = "62648d6"


def probe(source, output):
    """Exercise the actual engine with deterministic inference, without a GPU."""
    sys.path.insert(0, str(source))
    from locaii.engine import Engine
    from locaii.storage import Store

    class Model:
        """Repeat a page until recovery guidance, or perform 45 distinct reads."""

        def __init__(self, scenario):
            self.scenario = scenario
            self.rounds = 0

        def list_models(self):
            return [{"id": "fixture", "context": 32768}]

        def start(self, *args, **kwargs):
            pass

        def count_tokens(self, *args, **kwargs):
            return 100

        def chat(self, history, tools=None, **kwargs):
            if history[0]["content"].startswith("Write concise Markdown task memory"):
                return {
                    "message": {
                        "role": "assistant",
                        "content": "Synthetic verification only.",
                    }
                }
            self.rounds += 1
            if self.scenario == "progress":
                offset = (self.rounds - 1) * 256
                done = self.rounds > 45
            else:
                # A scripted cooperative model obeys an explicit pagination hint.
                guidance = any(
                    m.get("role") == "user"
                    and "next_offset_chars" in str(m.get("content"))
                    for m in history
                )
                offset = 256 if guidance else 0
                done = guidance and any(
                    m.get("role") == "tool" and "DATA42" in str(m.get("content"))
                    for m in history
                )
            message = {
                "role": "assistant",
                "content": (
                    "Verified the requested data."
                    if done
                    else "Inspecting the next data chunk."
                ),
            }
            if not done:
                message["tool_calls"] = [
                    {
                        "id": str(self.rounds),
                        "type": "function",
                        "function": {
                            "name": "read_text_file",
                            "arguments": json.dumps(
                                {
                                    "path": "fixture.txt",
                                    "offset_chars": offset,
                                    "max_chars": 256,
                                }
                            ),
                        },
                    }
                ]
            return {
                "message": message,
                "finish_reason": "stop" if done else "tool_calls",
                "usage": {},
            }

    rows = []
    for scenario in ("progress", "repetition"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Store(root / "state")
            session = store.create_session(
                "Synthetic loop benchmark", root / "workspace"
            )
            (root / "workspace/fixture.txt").write_text(
                "H" * 256 + "DATA42" + "abcdef" * 2000
            )
            model = Model(scenario)
            try:
                result = Engine(store, model).run(
                    session["id"],
                    "Inspect the data and explain the result.",
                    options={"model_id": "fixture", "supervisor_provider": "off"},
                )
                status = result["status"]
            except Exception as error:
                status = str(error)
            rows.append(
                {
                    "scenario": scenario,
                    "completed": int(status == "completed"),
                    "inference_rounds": model.rounds,
                    "status": status,
                }
            )
    output.write_text(json.dumps(rows, indent=2))


def main():
    """Archive the baseline privately, run both versions, and plot observed rounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(".local/verification/iteration-recovery")
    )
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe(args.probe, args.output / "probe.json")
        return
    repo = Path(__file__).resolve().parents[1]
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        archive = subprocess.check_output(
            ["git", "archive", args.baseline, "src"], cwd=repo
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(temporary, filter="data")
        for label, source in [
            ("before", Path(temporary) / "src"),
            ("after", repo / "src"),
        ]:
            out = args.output / label
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--probe",
                    str(source),
                    "--output",
                    str(out),
                ],
                check=True,
            )
            rows.extend(
                dict(version=label, **row)
                for row in json.loads((out / "probe.json").read_text())
            )
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), layout="constrained")
    for ax, scenario, title in zip(
        axes,
        ("progress", "repetition"),
        ("45 useful read steps", "Repeated first-page read"),
    ):
        selected = [row for row in rows if row["scenario"] == scenario]
        bars = ax.bar(
            ["Before", "After"],
            [row["inference_rounds"] for row in selected],
            width=0.35,
            color=["#94a3b8", "#0f766e"],
        )
        for bar, row in zip(bars, selected):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                "Completed" if row["completed"] else "Stopped",
                ha="center",
                fontsize=9,
            )
        ax.set_ylim(0, 55)
        ax.set_title(title)
        ax.set_ylabel("Inference rounds")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Deterministic regression comparison, not a model-quality benchmark",
        fontsize=10,
    )
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output / f"comparison.{extension}", dpi=180)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
