#!/usr/bin/env python3
"""Compare deterministic simple-chat behavior against a specified prior commit.

No provider or inference calls are made. Scripted local answers and a synthetic
workspace isolate review scheduling and allowance handling from model quality.
"""

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

OUTPUT = Path("docs/benchmarks")
FIGURE_SIZE = (8, 3.8)
COLORS = ["#a7b3ac", "#426c5b"]
RUNNER = """
import json,tempfile
from pathlib import Path
from locaii.engine import Engine
from locaii.storage import Store
from locaii.supervision import Supervisor
from test_engine import FakeModels, FakeSupervisor, answer, run
rows=[]
for case in ('simple_chat','insufficient_allowance'):
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);store=Store(root/'state');session=store.create_session('Synthetic',root/'workspace')
  supervisor=FakeSupervisor([{'verdict':'complete','summary':'Okay','next_steps':[]}]*3) if case=='simple_chat' else Supervisor(store)
  options={'supervisor_provider':'local'} if case=='simple_chat' else {'supervisor_provider':'codex-cli','remote_enabled':True,'supervisor_token_budget':1}
  try:
   result=run(Engine(store,FakeModels([answer('Hello.')]),supervisor),session['id'],'hello',options=options)
   status=result['status']
  except RuntimeError:
   status='error'
  rows.append({'case':case,'status':status,'completed':int(status=='completed'),'review_calls':supervisor.calls if case=='simple_chat' else ''})
print(json.dumps(rows))
"""


def main():
    """Run identical fixtures against archived and current engine code, then plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp:
        archive = subprocess.check_output(
            ["git", "archive", args.before, "src/locaii"], cwd=repo
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(temp, filter="data")
        rows = []
        for version, source in [
            ("before", Path(temp) / "src"),
            ("after", repo / "src"),
        ]:
            env = {
                **os.environ,
                "PYTHONPATH": str(source) + os.pathsep + str(repo / "tests"),
            }
            data = json.loads(
                subprocess.check_output(
                    [sys.executable, "-c", RUNNER], cwd=temp, env=env, text=True
                )
            )
            rows.extend({"version": version, **row} for row in data)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    with (OUTPUT / "review-scope.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    for ax, case, key, title in zip(
        axes,
        ["simple_chat", "insufficient_allowance"],
        ["review_calls", "completed"],
        [
            "Automatic reviews for a simple reply",
            "Local answer completes with no review budget",
        ],
    ):
        selected = [row for row in rows if row["case"] == case]
        ax.bar(
            ["Before", "After"],
            [row[key] for row in selected],
            width=0.35,
            color=COLORS,
        )
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_ylim(0, 2.4 if key == "review_calls" else 1.2)
        ax.set_yticks([0, 1, 2] if key == "review_calls" else [0, 1])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Deterministic regression comparison · identical synthetic chat",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.04,
        0.03,
        "Scripted answers, no inference or provider calls. Behavior checks, not timing or model-quality measurements.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.1, 1, 0.91])
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"review-scope.{extension}", dpi=170)
    plt.close(fig)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
