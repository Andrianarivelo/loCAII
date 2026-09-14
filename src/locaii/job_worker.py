"""Independent command worker. Durable receipts survive a closed GUI."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .storage import atomic_json


def sandbox_command(job):
    """Apply the task's write permission to shell access.

    With source writes enabled, the shell can modify files allowed by the
    operating-system account. Read-only tasks retain the restricted mounts.
    Network and process isolation apply to both modes.
    """
    executable = shutil.which("bwrap")
    if executable is None:
        raise RuntimeError(
            "Command refused: bubblewrap (bwrap) is required for terminal isolation. "
            "Install bubblewrap before running terminal commands."
        )
    roots = job.get("write_roots", [])
    if not isinstance(roots, list) or not all(isinstance(p, str) for p in roots):
        raise ValueError("Command refused: write_roots must be a list of directories.")
    source_writes = job.get("allow_source_writes") is True
    command = [
        executable,
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-net",
        "--cap-drop",
        "ALL",
        "--bind" if source_writes else "--ro-bind",
        "/",
        "/",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for value in roots:
        root = Path(value)
        if (
            not root.is_absolute()
            or not root.is_dir()
            or root == Path("/")
            or root.resolve() != root
        ):
            raise ValueError(
                f"Command refused: writable directory must exist, be absolute, "
                f"and contain no symbolic-link path components: {value}"
            )
        # A pre-existing hardlink could alias a source inode despite the
        # read-only source mount. Result directories must not contain these.
        for directory, _subdirectories, files in (
            () if source_writes else os.walk(root, followlinks=False)
        ):
            for filename in files:
                entry = Path(directory) / filename
                if not entry.is_symlink() and entry.stat().st_nlink > 1:
                    raise ValueError(
                        f"Command refused: writable results contain a hardlinked "
                        f"file that could modify source data: {entry}"
                    )
        command.extend(["--bind", str(root), str(root)])
    if roots:
        command.extend(
            [
                "--setenv",
                "LOCAII_OUTPUT_DIR",
                job.get("output_directory", roots[-1]),
                "--setenv",
                "LOCAII_WORK_DIR",
                roots[0],
                "--setenv",
                "TMPDIR",
                roots[0],
            ]
        )
    command.extend(["--setenv", "PYTHONDONTWRITEBYTECODE", "1"])
    command.extend(["--", "/bin/bash", "--noprofile", "--norc", "-c", job["command"]])
    return command


def main():
    """Run one user-enabled terminal job and atomically record its actual outcome."""
    path = Path(sys.argv[1])
    job = json.loads(path.read_text())
    atomic_json(
        path.with_suffix(".worker"),
        {
            "pid": os.getpid(),
            "process_start": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
        },
    )
    try:
        if path.with_suffix(".cancel").exists():
            job.update(status="cancelled", finished=time.time())
            atomic_json(path, job)
            return
        with (
            path.with_suffix(".stdout").open("wb") as out,
            path.with_suffix(".stderr").open("wb") as err,
        ):
            process = subprocess.Popen(
                sandbox_command(job),
                cwd=job["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
            job.update(status="running", pid=process.pid, started=time.time())
            job["process_start"] = (
                Path(f"/proc/{process.pid}/stat").read_text().split()[21]
            )
            atomic_json(path, job)
            try:
                deadline = time.monotonic() + job.get("timeout", 3600)
                while process.poll() is None:
                    if path.with_suffix(".cancel").exists():
                        import signal

                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        job["cancelled"] = True
                        break
                    if time.monotonic() > deadline:
                        raise subprocess.TimeoutExpired(
                            job["command"], job.get("timeout", 3600)
                        )
                    time.sleep(0.1)
                code = process.returncode
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
                job["error"] = "Command exceeded its time limit."
            if code and not job.get("error") and not job.get("cancelled"):
                # Preserve the command's diagnostic and add a concrete remedy
                # when the kernel rejected an attempted source-file write.
                err.flush()
                diagnostic = path.with_suffix(".stderr").read_text(errors="replace")
                if "Read-only file system" in diagnostic:
                    if job.get("allow_source_writes") is True:
                        job["error"] = (
                            "The operating system reported a read-only filesystem: "
                            + diagnostic.strip()
                        )
                    else:
                        job["error"] = (
                            "Source-file write refused: writes are disabled for this task. "
                            "Enable workspace writes to allow shell source edits."
                        )
                elif "bwrap:" in diagnostic:
                    job["error"] = (
                        "Command sandbox failed to start; the command was not "
                        "retried outside the configured isolation. "
                        + diagnostic.strip()
                    )
            job.update(
                status=(
                    "cancelled"
                    if job.get("cancelled")
                    else ("completed" if code == 0 else "failed")
                ),
                exit_code=code,
                finished=time.time(),
            )
    except Exception as error:
        job.update(status="failed", error=str(error), finished=time.time())
    atomic_json(path, job)


if __name__ == "__main__":
    main()
