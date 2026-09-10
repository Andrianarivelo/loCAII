"""Private SQLite transcripts and atomic task memory/checkpoints.

The database stores complete messages. Context reduction only changes a derived
prompt; it never deletes source history. Each write transaction is durable.
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid


def canonical(value) -> str:
    """Stable serialization makes matching independent of object key ordering."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    """Hash complete content, including the middle of very large tool receipts."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    """Publish private JSON only after all bytes have reached durable storage."""
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_text(path: Path, text: str) -> None:
    """Replace a regular managed file atomically without following symlinks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ValueError("Refusing to overwrite a symbolic link in task storage.")
    temporary = path.parent / ("." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    """Thread-safe task repository shared by the GUI and background workers."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "tasks.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,title TEXT,workspace TEXT,status TEXT,updated REAL);
        CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,payload TEXT,created REAL);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,type TEXT,payload TEXT,created REAL);
        """)
        with self.db:
            self.db.execute(
                "UPDATE sessions SET status='interrupted' WHERE status='running'"
            )

    def sessions(self) -> list[dict]:
        """Return newest tasks first without loading their private transcripts."""
        with self._lock:
            return [
                dict(row)
                for row in self.db.execute(
                    "SELECT * FROM sessions ORDER BY updated DESC"
                )
            ]

    def create_session(self, title: str, workspace: str | Path) -> dict:
        """Create a workspace-linked task and its initial Markdown memory."""
        workspace = Path(workspace).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        identifier = uuid.uuid4().hex
        with self._lock, self.db:
            self.db.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?)",
                (
                    identifier,
                    title[:140] or "New task",
                    str(workspace),
                    "ready",
                    time.time(),
                ),
            )
        self.save_memory(
            identifier, "# Task memory\n\nNo work has been performed yet.\n"
        )
        return self.get_session(identifier)

    def get_session(self, identifier: str) -> dict:
        """Resolve only existing task IDs, never arbitrary paths supplied by a model."""
        with self._lock:
            row = self.db.execute(
                "SELECT * FROM sessions WHERE id=?", (identifier,)
            ).fetchone()
        if not row:
            raise KeyError("Task not found.")
        return dict(row)

    def task_dir(self, identifier: str) -> Path:
        """Resolve a validated private task directory."""
        self.get_session(identifier)
        path = self.root / "tasks" / identifier
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path

    def set_status(self, identifier: str, status: str) -> None:
        """Persist visible state after completion, interruption, or a real failure."""
        with self._lock, self.db:
            self.db.execute(
                "UPDATE sessions SET status=?,updated=? WHERE id=?",
                (status, time.time(), identifier),
            )

    def append(self, identifier: str, role: str | dict, content="", **extra) -> int:
        """Record the complete native message before continuing to another operation."""
        self.get_session(identifier)
        message = (
            dict(role)
            if isinstance(role, dict)
            else {"role": role, "content": content, **extra}
        )
        with self._lock, self.db:
            cursor = self.db.execute(
                "INSERT INTO messages(session_id,payload,created) VALUES(?,?,?)",
                (identifier, canonical(message), time.time()),
            )
            self.db.execute(
                "UPDATE sessions SET updated=? WHERE id=?", (time.time(), identifier)
            )
            return cursor.lastrowid

    def messages(self, identifier: str) -> list[dict]:
        """Read ordered model-compatible messages with full tool arguments/results."""
        with self._lock:
            return [
                json.loads(row[0])
                for row in self.db.execute(
                    "SELECT payload FROM messages WHERE session_id=? ORDER BY id",
                    (identifier,),
                )
            ]

    def event(self, identifier: str, kind: str, payload: dict) -> None:
        """Write structured activity and validation receipts separately from the prompt."""
        with self._lock, self.db:
            self.db.execute(
                "INSERT INTO events(session_id,type,payload,created) VALUES(?,?,?,?)",
                (identifier, kind, canonical(payload), time.time()),
            )

    def events(self, identifier: str, limit=100) -> list[dict]:
        """Return a bounded chronological activity tail for inspection/supervision."""
        with self._lock:
            rows = self.db.execute(
                "SELECT type,payload,created FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (identifier, max(1, min(1000, limit))),
            ).fetchall()
        return [
            {
                "type": row["type"],
                **json.loads(row["payload"]),
                "created": row["created"],
            }
            for row in reversed(rows)
        ]

    def memory(self, identifier: str) -> str:
        """Read fallible notes, never interpret them as new user authorization."""
        path = self.task_dir(identifier) / "MEMORY.md"
        return path.read_text() if path.exists() else ""

    def supervisor_tokens(self, identifier: str, remote_only: bool = False) -> int:
        """Count the whole task's review usage, independent of activity-tail limits."""
        with self._lock:
            rows = self.db.execute(
                "SELECT payload FROM events WHERE session_id=? AND type='supervisor_usage'",
                (identifier,),
            ).fetchall()
        usage = [json.loads(row[0]) for row in rows]
        return sum(
            int(item.get("charged_tokens", 0))
            for item in usage
            if not remote_only or item.get("provider") != "local"
        )

    def save_memory(self, identifier: str, text: str) -> Path:
        """Retain an archive receipt outside the model-controlled bounded notes."""
        checkpoint = self.checkpoint(identifier)
        receipt = ""
        if checkpoint.get("archive"):
            receipt = (
                "\n\n## Full context archive\n"
                + checkpoint["archive"]
                + "\nUse read_context_archive for bounded historical pages.\n"
            )
        path = self.task_dir(identifier) / "MEMORY.md"
        if path.exists():
            atomic_text(path.with_name("MEMORY.previous.md"), path.read_text())
        atomic_text(path, text[: max(0, 16000 - len(receipt))] + receipt)
        return path

    def checkpoint(self, identifier: str) -> dict:
        """Reject malformed checkpoints instead of using incomplete summaries."""
        path = self.task_dir(identifier) / "checkpoint.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            return {}

    def save_checkpoint(self, identifier: str, value: dict) -> None:
        """Commit a cursor only after the matching archive exists."""
        atomic_json(self.task_dir(identifier) / "checkpoint.json", value)

    def archive(self, identifier: str, messages: list[dict]) -> Path:
        """Preserve full history immutably before evicting anything from active context."""
        value = {
            "version": 1,
            "messages": messages,
            "memory": self.memory(identifier),
            "created": time.time(),
        }
        encoded = canonical(value)
        if len(encoded.encode()) > 64 * 1024 * 1024:
            raise ValueError(
                "Context archive exceeds 64 MiB; complete history remains in the database."
            )
        path = self.task_dir(identifier) / ("context-" + uuid.uuid4().hex + ".json")
        atomic_json(path, value)
        return path

    def close(self) -> None:
        """Flush the database when a process explicitly shuts down."""
        with self._lock:
            self.db.close()
