"""Real subprocess regressions for accidental source-data destruction.

Only disposable pytest files are touched. Commands run through the same durable
worker used by the desktop app, so these checks exercise kernel mount isolation
rather than merely asserting that a sandbox flag was added to an argument list.
"""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import pytest

from locaii.job_worker import sandbox_command


@pytest.fixture
def command_job(tmp_path):
    """Build a raw input, isolated result directory, and a worker launcher."""
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap is required for kernel sandbox integration checks")
    original = tmp_path / "metadata.csv"
    contents = "mouse_1_center_x,mouse_2_center_x\n1,2\n" * 300
    original.write_text(contents)
    output = tmp_path / "results"
    output.mkdir()

    def launch(command, *, roots=None, timeout=10, source_writes=False):
        """Run a disposable durable job and return the exact worker receipt."""
        path = tmp_path / "job.json"
        path.write_text(
            json.dumps(
                {
                    "command": command,
                    "cwd": str(tmp_path),
                    "timeout": timeout,
                    "write_roots": [str(output)] if roots is None else roots,
                    "allow_source_writes": source_writes,
                }
            )
        )
        subprocess.run(
            [sys.executable, "-m", "locaii.job_worker", str(path)],
            check=True,
            timeout=20,
        )
        receipt = json.loads(path.read_text())
        receipt["stderr"] = path.with_suffix(".stderr").read_text()
        receipt["stdout"] = path.with_suffix(".stdout").read_text()
        return receipt

    return original, contents, output, launch


def test_command_cannot_truncate_original_metadata(command_job):
    """Reproduce the real read-5000-then-overwrite failure on disposable data."""
    original, contents, _output, launch = command_job
    script = (
        "from pathlib import Path; "
        f"p = Path({str(original)!r}); "
        "preview = p.read_text()[:5000]; p.write_text(preview)"
    )
    receipt = launch(f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}")
    assert receipt["status"] == "failed", receipt
    assert "Source-file write refused" in receipt["error"]
    assert original.read_text() == contents


def test_command_can_read_source_and_save_derived_output(command_job):
    """Read-only source mounts still support ordinary analysis workflows."""
    original, contents, output, launch = command_job
    target = output / "derived.csv"
    receipt = launch(
        f"head -n 2 {shlex.quote(str(original))} > {shlex.quote(str(target))}"
    )
    assert receipt["status"] == "completed", receipt
    assert target.read_text() == "mouse_1_center_x,mouse_2_center_x\n1,2\n"
    assert original.read_text() == contents


def test_symlink_inside_results_cannot_escape_to_source(command_job):
    original, contents, output, launch = command_job
    link = output / "innocent-looking.csv"
    link.symlink_to(original)
    receipt = launch(f"printf corrupted > {shlex.quote(str(link))}")
    assert receipt["status"] == "failed", receipt
    assert original.read_text() == contents


def test_hardlink_inside_results_is_refused_before_execution(command_job):
    original, contents, output, launch = command_job
    link = output / "hardlink.csv"
    os.link(original, link)
    receipt = launch(f"printf corrupted > {shlex.quote(str(link))}")
    assert receipt["status"] == "failed", receipt
    assert "hardlinked" in receipt["error"]
    assert original.read_text() == contents


def test_missing_write_roots_leave_entire_host_read_only(command_job):
    original, contents, output, launch = command_job
    receipt = launch(
        f"printf corrupted > {shlex.quote(str(output / 'new.csv'))}", roots=[]
    )
    assert receipt["status"] == "failed", receipt
    assert not (output / "new.csv").exists()
    assert original.read_text() == contents


def test_missing_sandbox_fails_closed(monkeypatch, tmp_path):
    """No fallback may execute a shell directly when protection is absent."""
    monkeypatch.setattr("locaii.job_worker.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="bubblewrap.*required"):
        sandbox_command({"command": "touch destroyed", "cwd": str(tmp_path)})
    assert not (tmp_path / "destroyed").exists()


def test_symbolic_link_write_root_is_refused(command_job, tmp_path):
    original, contents, output, launch = command_job
    alias = tmp_path / "result-alias"
    alias.symlink_to(output, target_is_directory=True)
    receipt = launch("printf unsafe", roots=[str(alias)])
    assert receipt["status"] == "failed", receipt
    assert "symbolic-link" in receipt["error"]
    assert original.read_text() == contents


def test_sandboxed_command_timeout_is_preserved(command_job):
    _original, _contents, _output, launch = command_job
    started = time.monotonic()
    receipt = launch("sleep 30", timeout=0.1)
    assert receipt["status"] == "failed", receipt
    assert receipt["exit_code"] == 124
    assert time.monotonic() - started < 8


def test_cancel_marker_stops_sandbox_and_children(tmp_path):
    """An active command remains cancellable after entering its PID namespace."""
    if not shutil.which("bwrap"):
        pytest.skip("bubblewrap is required for kernel sandbox integration checks")
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "command": "sleep 30 & wait",
                "cwd": str(tmp_path),
                "timeout": 15,
            }
        )
    )
    worker = subprocess.Popen([sys.executable, "-m", "locaii.job_worker", str(path)])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            receipt = json.loads(path.read_text())
            if receipt.get("status") == "running":
                break
            time.sleep(0.02)
        assert receipt.get("status") == "running", receipt
        path.with_suffix(".cancel").touch()
        worker.wait(timeout=8)
        receipt = json.loads(path.read_text())
        assert receipt["status"] == "cancelled", receipt
        assert not Path(f"/proc/{receipt['pid']}").exists()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


@pytest.mark.parametrize("link_kind", [None, "symlink", "hardlink"])
def test_enabled_source_writes_modify_existing_source(command_job, link_kind):
    """Explicit write permission permits real edits, including ordinary file links."""
    original, _contents, output, launch = command_job
    target = original
    if link_kind:
        target = output / "source-link.csv"
        if link_kind == "symlink":
            target.symlink_to(original)
        else:
            os.link(original, target)
    receipt = launch(
        f"printf 'x,y\\n3,4\\n' > {shlex.quote(str(target))}", source_writes=True
    )
    assert receipt["status"] == "completed", receipt
    assert original.read_text() == "x,y\n3,4\n"
