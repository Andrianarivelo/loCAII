"""Session-only CLI launch configuration, with no provider calls or authentication reads."""

import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from locaii.supervisor_launch import SupervisorLauncher, terminal_command
import locaii.supervisor_launch as launch


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Use hostile-looking paths as literal filenames and a fake CLI config listing."""
    root = tmp_path / "workspace ' $(touch SHOULD_NOT_EXIST)"
    root.mkdir()
    connection = root / "connection.json"
    connection.write_text(json.dumps({"token": "PRIVATE_CREDENTIAL_1234567890"}))
    monkeypatch.setattr(launch, "_binary", lambda name: "/fake/bin/" + name)
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=json.dumps([{"name": "other.server"}, {"name": "locaii"}]),
            returncode=0,
        ),
    )
    return root, connection


def test_codex_launch_is_scoped_and_uses_existing_provider_defaults(prepared):
    root, connection = prepared
    result = SupervisorLauncher(root).prepare(
        "codex", connection, str(root), {"id": "qwen-test", "password": "SECRET"}
    )
    command = result["command"]
    overrides = [
        command[i + 1] for i, value in enumerate(command[:-1]) if value == "-c"
    ]
    assert any(
        '"other.server" = { enabled = false, command = "false" }' in value
        for value in overrides
    )
    assert any(
        value.startswith("mcp_servers=") and "delegation_mcp" in value
        for value in overrides
    )
    assert "--model" not in command and "--ignore-user-config" not in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "plugins" in command and "hooks" in command
    assert "PRIVATE_CREDENTIAL" not in json.dumps(result)
    assert "SECRET" not in Path(result["guide_path"]).read_text()
    assert "subordinate" in Path(result["guide_path"]).read_text()
    assert Path(result["launch_file"]).stat().st_mode & 0o777 == 0o700
    script = Path(result["launch_file"]).read_text().splitlines()
    argv = shlex.split(next(line[5:] for line in script if line.startswith("exec ")))
    assert argv[-1] == str(Path(result["cwd"]) / "command.json")
    assert json.loads(Path(argv[-1]).read_text())["command"] == command


def test_claude_has_only_local_bridge_and_no_builtin_tools(prepared):
    root, connection = prepared
    result = SupervisorLauncher(root).prepare(
        "claude-cli", connection, str(root), {"id": "qwen"}
    )
    command = result["command"]
    config = json.loads(Path(command[command.index("--mcp-config") + 1]).read_text())
    assert list(config["mcpServers"]) == ["locaii"]
    assert "--strict-mcp-config" in command
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--allowedTools") + 1] == "mcp__locaii__*"
    assert str(connection) in config["mcpServers"]["locaii"]["args"]


def test_config_listing_failure_is_actionable_and_redacted(prepared, monkeypatch):
    root, connection = prepared

    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="PRIVATE_CREDENTIAL")

    monkeypatch.setattr(launch.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="inspect Codex MCP") as error:
        SupervisorLauncher(root).prepare("codex", connection, str(root), {})
    assert "PRIVATE_CREDENTIAL" not in str(error.value)


def test_terminal_preserves_literal_argv(monkeypatch):
    monkeypatch.setattr(
        launch.shutil,
        "which",
        lambda name: "/usr/bin/ptyxis" if name == "ptyxis" else None,
    )
    command = ["/tmp/a ' $(false)/launch.sh"]
    result = terminal_command(command, "/tmp/work space")
    assert result[-2:] == ["--", command[0]]
    assert result[result.index("--working-directory") + 1] == "/tmp/work space"


def test_missing_terminal_provides_copy_fallback(monkeypatch):
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Copy connection command"):
        terminal_command(["/tmp/launch.sh"], "/tmp")
