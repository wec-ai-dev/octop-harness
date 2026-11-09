"""Tests for the OpenSandbox backend (mocked SDK — no live server)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from octop_harness.backends import BackendWorkspace, resolve_backend, spec_supports_execution
from octop_harness.backends.opensandbox_sandbox import OpenSandbox


def _msg(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def _execution(*, stdout: str = "ok", stderr: str = "", exit_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        logs=SimpleNamespace(
            stdout=[_msg(stdout)] if stdout else [],
            stderr=[_msg(stderr)] if stderr else [],
        ),
        exit_code=exit_code,
    )


@pytest.fixture
def mock_opensandbox() -> Any:
    sandbox = MagicMock()
    sandbox.id = "osb-test-id"
    sandbox.commands.run.return_value = _execution()
    sandbox.files.write_files.return_value = None
    sandbox.files.read_bytes.return_value = b"hello"
    sandbox.destroy.return_value = None

    conn_cls = MagicMock()
    sandbox_sync = MagicMock()
    sandbox_sync.create.return_value = sandbox

    write_entry = MagicMock()
    run_opts = MagicMock()

    modules = {
        "opensandbox": MagicMock(SandboxSync=sandbox_sync),
        "opensandbox.config.connection_sync": MagicMock(ConnectionConfigSync=conn_cls),
        "opensandbox.models.filesystem": MagicMock(WriteEntry=write_entry),
        "opensandbox.models.execd": MagicMock(RunCommandOpts=run_opts),
    }
    with patch.dict("sys.modules", modules):
        yield sandbox_sync, sandbox, conn_cls


class TestSpecSupportsExecution:
    def test_opensandbox_string(self) -> None:
        assert spec_supports_execution("opensandbox") is True

    def test_opensandbox_dict(self) -> None:
        assert spec_supports_execution({"type": "opensandbox", "image": "python:3.12"}) is True


class TestMissingDependency:
    def test_import_error_mentions_extra(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "opensandbox" or name.startswith("opensandbox."):
                raise ImportError("No module named opensandbox")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_fake_import),
            pytest.raises(ImportError, match=r"octop-harness\[opensandbox\]"),
        ):
            resolve_backend({"type": "opensandbox", "domain": "localhost:8080"})


class TestLifecycle:
    def test_resolve_creates_and_close_destroys(self, mock_opensandbox: Any) -> None:
        sandbox_sync, sandbox, conn_cls = mock_opensandbox
        backend = resolve_backend(
            {
                "type": "opensandbox",
                "image": "python:3.12",
                "api_key": "k",
                "domain": "localhost:8080",
                "protocol": "http",
            }
        )
        sandbox_sync.create.assert_called_once()
        conn_cls.assert_called_once()
        result = backend.execute("echo hi")
        assert result.exit_code == 0
        assert "ok" in result.output
        backend.close()
        sandbox.destroy.assert_called_once()
        backend.close()
        assert sandbox.destroy.call_count == 1

    def test_upload_and_download(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080"})
        uploaded = backend.upload_files([("/tmp/a.txt", b"abc")])
        assert uploaded[0].error is None
        sandbox.files.write_files.assert_called_once()
        downloaded = backend.download_files(["/tmp/a.txt"])
        assert downloaded[0].content == b"hello"
        rejected = backend.upload_files([("relative.txt", b"x")])
        assert rejected[0].error is not None

    def test_mkdir_path_quotes_the_path(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080"})
        backend.mkdir_path("/workspace/a b;rm -rf /")
        command = sandbox.commands.run.call_args.args[0]
        assert command == "mkdir -p '/workspace/a b;rm -rf /'"

    def test_mkdir_path_nonzero_exit_raises(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        sandbox.commands.run.return_value = _execution(stdout="denied", exit_code=1)
        backend = resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080"})
        with pytest.raises(OSError, match="denied"):
            backend.mkdir_path("/workspace/skills")

    def test_delete_path_quotes_the_path(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080"})
        backend.delete_path("/skills/a b;rm -rf /")
        command = sandbox.commands.run.call_args.args[0]
        assert command == "rm -rf '/skills/a b;rm -rf /'"

    def test_delete_path_nonzero_exit_raises(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        sandbox.commands.run.return_value = _execution(stdout="busy", exit_code=1)
        backend = resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080"})
        with pytest.raises(OSError, match="busy"):
            backend.delete_path("/skills/demo")

    def test_move_path_creates_parent_and_quotes(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080"})
        backend.move_path("/old;rm", "/new name/file;x")
        command = sandbox.commands.run.call_args.args[0]
        assert command == "mkdir -p '/new name' && mv '/old;rm' '/new name/file;x'"

    def test_move_path_nonzero_exit_raises(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        sandbox.commands.run.return_value = _execution(stdout="missing", exit_code=1)
        backend = resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080"})
        with pytest.raises(OSError, match="missing"):
            backend.move_path("/skills/old", "/skills/new")


class TestSandboxWorkspaceDir:
    """``sandbox_workspace_dir`` keeps the agent workspace out of the sandbox root."""

    @staticmethod
    def _run_opts() -> MagicMock:
        return sys.modules["opensandbox.models.execd"].RunCommandOpts  # type: ignore[no-any-return]

    def test_default_has_no_directory_and_runs_commands_as_before(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080"})
        assert backend.sandbox_workspace_dir == ""
        backend.execute("echo hi")
        assert self._run_opts().call_args.kwargs["working_directory"] is None
        assert sandbox.commands.run.call_count == 1

    def test_commands_start_in_the_directory_created_once(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = resolve_backend(
            {"type": "opensandbox", "domain": "127.0.0.1:8080", "sandbox_workspace_dir": "/workspace/"}
        )
        assert backend.sandbox_workspace_dir == "/workspace"
        assert backend.execute("ls").exit_code == 0
        assert backend.execute("pwd").exit_code == 0
        calls = sandbox.commands.run.call_args_list
        assert len(calls) == 3
        assert calls[0].args == ("mkdir -p /workspace",)
        assert "opts" not in calls[0].kwargs
        assert [call.args[0] for call in calls[1:]] == ["ls", "pwd"]
        assert self._run_opts().call_args.kwargs["working_directory"] == "/workspace"

    def test_injected_sandbox_gets_the_same_behaviour(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = OpenSandbox(sandbox=sandbox, sandbox_workspace_dir="/workspace")
        backend.execute("ls")
        assert sandbox.commands.run.call_args_list[0].args == ("mkdir -p /workspace",)
        assert self._run_opts().call_args.kwargs["working_directory"] == "/workspace"

    def test_directory_failure_is_reported_and_retried(self, mock_opensandbox: Any) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = resolve_backend(
            {"type": "opensandbox", "domain": "127.0.0.1:8080", "sandbox_workspace_dir": "/workspace"}
        )
        sandbox.commands.run.return_value = _execution(stdout="", exit_code=1)
        failed = backend.execute("ls")
        assert failed.exit_code == 1
        assert "/workspace" in failed.output
        assert sandbox.commands.run.call_count == 1
        sandbox.commands.run.return_value = _execution()
        assert backend.execute("ls").exit_code == 0

    def test_relative_directory_is_rejected(self, mock_opensandbox: Any) -> None:
        with pytest.raises(ValueError, match="absolute"):
            resolve_backend({"type": "opensandbox", "domain": "127.0.0.1:8080", "sandbox_workspace_dir": "workspace"})

    def test_backend_workspace_maps_into_the_directory(self, mock_opensandbox: Any, tmp_path: Path) -> None:
        _sync, sandbox, _conn = mock_opensandbox
        backend = resolve_backend(
            {"type": "opensandbox", "domain": "127.0.0.1:8080", "sandbox_workspace_dir": "/workspace"}
        )
        workspace = BackendWorkspace(backend, tmp_path)
        assert workspace.resolve_path("notes/a.md") == "/workspace/notes/a.md"
        workspace.mkdir("notes")
        assert sandbox.commands.run.call_args.args[0] == "mkdir -p /workspace/notes"
