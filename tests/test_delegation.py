"""Real loopback and stdio protocol tests, with no GPU or paid supervisor calls."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from locaii.delegation import DelegationHub, MAX_BODY, MAX_QUEUE
from locaii.delegation_mcp import BridgeClient, handle


@pytest.fixture
def bridge(tmp_path):
    hub = DelegationHub(tmp_path)
    path = hub.connect(
        str(tmp_path / "workspace"),
        {
            "supervisor_provider": "codex-cli",
            "remote_enabled": True,
            "shell_enabled": False,
        },
        {"id": "synthetic-local", "context": 32768},
    )
    yield hub, BridgeClient(path), path
    hub.disconnect()


def test_scope_permissions_and_connection_privacy(bridge):
    hub, client, path = bridge
    assert path.stat().st_mode & 0o777 == 0o600
    info = client.call("local_model_info")
    assert info["permissions"]["supervisor_provider"] == "off"
    assert info["permissions"]["remote_enabled"] is False
    assert info["permissions"]["shell_enabled"] is False
    assert info["model"]["id"] == "synthetic-local"
    assert "subordinate" in info["instructions"]
    for arguments in (
        {"task": "test", "request_id": "a", "options": {}},
        {"task": "test", "request_id": "a", "workspace": "/"},
    ):
        with pytest.raises(ValueError, match="unsupported"):
            client.call("delegate_task", arguments)
    job = client.call(
        "delegate_task", {"task": "Read the synthetic fixture", "request_id": "a"}
    )
    assert "options" not in job
    private_job = hub.take_next()
    private_job["options"]["shell_enabled"] = True
    assert hub._jobs[job["id"]]["options"]["shell_enabled"] is False
    assert hub.status()["last_client_seen"] is not None
    assert "token" not in hub.status()


def test_authentication_and_body_limit(bridge):
    _, _, path = bridge
    connection = json.loads(path.read_text())
    request = Request(
        connection["endpoint"], data=b"{}", headers={"Authorization": "Bearer wrong"}
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 401
    request = Request(
        connection["endpoint"],
        data=b"x" * (MAX_BODY + 1),
        headers={"Authorization": "Bearer " + connection["token"]},
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 413


def test_idempotency_concurrency_and_serial_execution(bridge):
    hub, client, _ = bridge
    arguments = {"task": "Write a fixture", "request_id": "same-request"}
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = list(
            executor.map(lambda _: client.call("delegate_task", arguments), range(4))
        )
    assert len({job["id"] for job in jobs}) == 1
    with pytest.raises(ValueError, match="different arguments"):
        client.call("delegate_task", {**arguments, "task": "Different work"})
    assert hub.take_next()["id"] == jobs[0]["id"]
    assert hub.take_next() is None
    assert hub.status()["active"]["id"] == jobs[0]["id"]
    hub.complete(jobs[0]["id"], {"status": "completed", "text": "done"})
    replay = client.call("delegate_task", arguments)
    assert replay["status"] == "completed"
    assert hub.take_next() is None


def test_receipts_usage_continue_and_no_session_overlap(bridge):
    hub, client, _ = bridge
    first = client.call(
        "delegate_task", {"task": "Read synthetic file", "request_id": "first"}
    )
    hub.take_next()
    hub.bind_session(first["id"], "session-1")
    hub.record_event(
        first["id"],
        "tool_result",
        {"name": "read_file", "text": "BIRCH-629", "api_key": "sensitive"},
    )
    hub.record_event(
        first["id"], "usage", {"prompt_tokens": 100, "completion_tokens": 20}
    )
    hub.record_event(first["id"], "usage", {"total_tokens": 42})
    with pytest.raises(ValueError, match="terminal"):
        client.call(
            "continue_task",
            {"job_id": first["id"], "task": "Check file", "request_id": "second"},
        )
    hub.complete(first["id"], {"status": "completed", "text": "BIRCH-629 verified"})
    receipt = client.call("task_status", {"job_id": first["id"]})
    assert receipt["local_model_tokens"] == 162
    assert (
        next(
            item["payload"]["text"]
            for item in receipt["events"]
            if item["event"] == "tool_result"
        )
        == "BIRCH-629"
    )
    assert "sensitive" not in json.dumps(receipt)
    followup = client.call(
        "continue_task",
        {"job_id": first["id"], "task": "Check file", "request_id": "second"},
    )
    assert followup["id"] != first["id"]
    assert followup["session_id"] == "session-1"
    same = client.call(
        "continue_task",
        {"job_id": first["id"], "task": "Check file", "request_id": "second"},
    )
    assert same["id"] == followup["id"]
    with pytest.raises(ValueError, match="already has"):
        client.call(
            "continue_task",
            {"job_id": first["id"], "task": "Again", "request_id": "third"},
        )


def test_cancellation_queue_bounds_and_disconnect(bridge):
    hub, client, path = bridge
    active = client.call(
        "delegate_task", {"task": "Run synthetic job", "request_id": "active"}
    )
    hub.take_next()
    queued = [
        client.call("delegate_task", {"task": "Queued fixture", "request_id": f"q{n}"})
        for n in range(MAX_QUEUE)
    ]
    with pytest.raises(ValueError, match="full"):
        client.call("delegate_task", {"task": "Too many", "request_id": "overflow"})
    cancelled = client.call("cancel_task", {"job_id": queued[0]["id"]})
    assert cancelled["status"] == "cancelled"
    hub.disconnect()
    assert not path.exists()
    assert hub._jobs[active["id"]]["status"] == "running"
    assert not hub.cancel_requested(active["id"])
    assert all(hub._jobs[job["id"]]["status"] == "cancelled" for job in queued)
    assert hub.take_next() is None
    with pytest.raises(ValueError, match="disconnected"):
        client.call("local_model_info")
    hub.complete(active["id"], {"status": "completed"})


def test_active_cancel_is_cooperative(bridge):
    hub, client, _ = bridge
    job = client.call("delegate_task", {"task": "fixture", "request_id": "cancel"})
    hub.take_next()
    receipt = client.call("cancel_task", {"job_id": job["id"]})
    assert receipt["status"] == "running"
    assert hub.cancel_requested(job["id"])
    hub.complete(job["id"], {"status": "interrupted", "text": "Progress saved"})
    assert client.call("task_status", {"job_id": job["id"]})["status"] == "interrupted"


def test_restart_interrupts_unknown_work_and_connection_ownership(tmp_path):
    hub = DelegationHub(tmp_path)
    client = BridgeClient(hub.connect(str(tmp_path), {}, {}))
    active = client.call("delegate_task", {"task": "fixture", "request_id": "first"})
    hub.take_next()
    # Simulate persisted crash-state discovery in a new process. The original
    # endpoint is closed only after the recovery assertion, as disconnect has
    # intentionally different semantics for queued work.
    recovered = DelegationHub(tmp_path)
    assert recovered._jobs[active["id"]]["status"] == "interrupted"
    hub.disconnect()
    new_client = BridgeClient(recovered.connect(str(tmp_path), {}, {}))
    try:
        with pytest.raises(ValueError, match="Unknown task"):
            new_client.call("task_status", {"job_id": active["id"]})
        assert recovered.take_next() is None
    finally:
        recovered.disconnect()


def test_status_wait_releases_lock_and_sees_real_completion(bridge):
    hub, client, _ = bridge
    job = client.call("delegate_task", {"task": "fixture", "request_id": "wait"})
    hub.take_next()
    start = time.monotonic()
    worker = threading.Thread(
        target=lambda: (
            time.sleep(0.15),
            hub.complete(job["id"], {"status": "completed", "text": "actual result"}),
        )
    )
    worker.start()
    result = client.call("task_status", {"job_id": job["id"], "wait_seconds": 5})
    worker.join()
    assert result["status"] == "completed"
    assert result["revision"] >= 3
    assert time.monotonic() - start < 2
    with pytest.raises(ValueError, match="wait_seconds"):
        client.call("task_status", {"job_id": job["id"], "wait_seconds": 26})


def test_mcp_stdio_process_and_tool_roundtrip(bridge):
    hub, _, path = bridge
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "delegate_task",
                "arguments": {"task": "Check synthetic receipt", "request_id": "stdio"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "shell", "arguments": {}},
        },
    ]
    result = subprocess.run(
        [sys.executable, "-m", "locaii.delegation_mcp", "--connection", str(path)],
        input="".join(json.dumps(message) + "\n" for message in messages),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 4
    assert responses[0]["result"]["protocolVersion"] == "2025-03-26"
    assert len(responses[1]["result"]["tools"]) == 5
    assert responses[2]["result"]["isError"] is False
    assert responses[3]["result"]["isError"] is True
    assert hub.take_next()["request_id"] == "stdio"
    assert json.loads(path.read_text())["token"] not in result.stdout + result.stderr


def test_malformed_mcp_messages_and_remote_endpoint_rejected(tmp_path):
    assert handle([], None)["error"]["code"] == -32600
    assert (
        handle({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": []}, None)[
            "error"
        ]["code"]
        == -32602
    )
    connection = tmp_path / "bad.json"
    connection.write_text(
        json.dumps({"endpoint": "https://example.com/rpc", "token": "a" * 40})
    )
    with pytest.raises(ValueError, match="loopback"):
        BridgeClient(connection)


def test_large_nested_evidence_is_bounded_and_latest_result_survives(bridge):
    hub, client, _ = bridge
    job = client.call("delegate_task", {"task": "fixture", "request_id": "large"})
    hub.take_next()
    nested = [{"text": "A" * 20000} for _ in range(100)]
    for index in range(30):
        hub.record_event(job["id"], "tool_result", {"index": index, "output": nested})
    hub.complete(job["id"], {"status": "completed", "text": "Latest verified result"})
    receipt = client.call("task_status", {"job_id": job["id"]})
    assert receipt["result"]["text"] == "Latest verified result"
    assert receipt["events"][0]["payload"]["index"] == 29
    assert len(json.dumps(receipt)) < 35000


def test_unknown_worker_status_never_reports_success(bridge):
    hub, client, _ = bridge
    job = client.call("delegate_task", {"task": "fixture", "request_id": "unknown"})
    hub.take_next()
    hub.complete(job["id"], {"status": "blocked", "text": "Could not inspect output"})
    receipt = client.call("task_status", {"job_id": job["id"]})
    assert receipt["status"] == "failed"
    assert receipt["result"]["status"] == "blocked"
    assert "unrecognized" in receipt["error"]


def test_old_longpoll_is_revoked_even_if_reconnect_occurs_before_wakeup(
    bridge, tmp_path
):
    hub, client, _ = bridge
    job = client.call("delegate_task", {"task": "fixture", "request_id": "old-poll"})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.call, "task_status", {"job_id": job["id"], "wait_seconds": 5}
        )
        time.sleep(0.1)
        # Hold the reentrant state lock across replacement so the old waiter
        # cannot observe the temporary token=None state. Ownership must still
        # be checked after the new connection becomes available.
        with hub._lock:
            hub.disconnect()
            hub.connect(str(tmp_path), {}, {})
        with pytest.raises(ValueError, match="revoked"):
            future.result(timeout=3)
