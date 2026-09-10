"""Run one real tool-free Codex review of a tiny synthetic task packet."""

from pathlib import Path
import json
import threading
from locaii.storage import Store
from locaii.supervision import Supervisor

ROOT = Path(__file__).resolve().parents[1]
store = Store(ROOT / ".local/verification/supervisor")
task = store.create_session(
    "Synthetic supervisor check", ROOT / ".local/verification/supervisor-workspace"
)
store.append(
    task["id"],
    "user",
    "Create a file with READY and read it back. Do not claim completion without a read receipt.",
)
store.append(
    task["id"], {"role": "assistant", "content": "I propose writing the file."}
)
review = Supervisor(store).review(
    task["id"],
    {
        "supervisor_provider": "codex-cli",
        "remote_enabled": True,
        "supervisor_token_budget": 30000,
    },
    None,
    threading.Event(),
)
result = {
    "review": review,
    "passed": review["verdict"] != "complete" and bool(review["next_steps"]),
    "events": store.events(task["id"]),
}
(ROOT / ".local/verification/supervisor-check.json").write_text(
    json.dumps(result, indent=2)
)
print(json.dumps(result))
if not result["passed"]:
    raise SystemExit(1)
