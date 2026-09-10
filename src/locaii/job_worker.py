"""Independent command worker. Durable receipts survive a closed GUI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .storage import atomic_json


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
                job["command"],
                shell=True,
                executable="/bin/bash",
                cwd=job["cwd"],
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
