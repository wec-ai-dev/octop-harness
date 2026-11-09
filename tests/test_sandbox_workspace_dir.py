"""``sandbox_workspace_dir``: a sandbox keeps its agent workspace in a directory of its own."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents.backends.protocol import LsResult, ReadResult

from octop_harness.backends.utils import normalize_sandbox_workspace_dir, sandbox_workspace_dir_of
from octop_harness.backends.workspace import BackendWorkspace


class _SandboxStub:
    """Sandbox backend that serves files by their literal sandbox path."""

    sandbox_fs = True
    virtual_mode = True

    def __init__(self, files: dict[str, str], *, workspace_dir: str | None = None) -> None:
        self.files = files
        if workspace_dir is not None:
            self.sandbox_workspace_dir = workspace_dir

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if file_path not in self.files:
            return ReadResult(error=f"File '{file_path}' not found")
        return ReadResult(file_data={"content": self.files[file_path], "encoding": "utf-8"})

    def ls(self, path: str) -> LsResult:
        prefix = path.rstrip("/") + "/"
        names = sorted({p for p in self.files if p.startswith(prefix)})
        return LsResult(entries=[{"path": name, "is_dir": False} for name in names])


def _workspace(tmp_path: Path, backend: Any, *, system_files_path: str | None = None) -> BackendWorkspace:
    return BackendWorkspace(backend, tmp_path, system_files_path=system_files_path)


class TestNormalize:
    @pytest.mark.parametrize("value", [None, "", "  ", "/", "//"])
    def test_no_directory(self, value: str | None) -> None:
        assert normalize_sandbox_workspace_dir(value) == ""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("/workspace", "/workspace"), ("/workspace/", "/workspace"), ("//work/./space", "/work/space")],
    )
    def test_absolute_paths_are_normalized(self, value: str, expected: str) -> None:
        assert normalize_sandbox_workspace_dir(value) == expected

    def test_relative_path_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="absolute"):
            normalize_sandbox_workspace_dir("workspace")

    def test_backend_without_the_attribute_has_no_directory(self) -> None:
        assert sandbox_workspace_dir_of(_SandboxStub({})) == ""
        assert sandbox_workspace_dir_of(None) == ""
        assert sandbox_workspace_dir_of(_SandboxStub({}, workspace_dir="/workspace")) == "/workspace"


class TestWithoutDirectory:
    """The sandbox root stays the workspace, as before."""

    def test_relative_paths_map_to_the_root(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, _SandboxStub({}), system_files_path=".octop")
        assert ws.resolve_path("SOUL.md") == "/SOUL.md"
        assert ws.resolve_path(".") == "/"
        assert ws.resolve_path("skills/demo") == "/.octop/skills/demo"
        assert ws.resolve_path("/skills/demo") == "/.octop/skills/demo"

    def test_listed_paths_lose_only_the_root_slash(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, _SandboxStub({"/a/b.md": "x"}))
        assert ws.present_path("/a/b.md") == "a/b.md"
        assert [row["path"] for row in ws.list_dir("a") or []] == ["a/b.md"]


class TestWithDirectory:
    def test_relative_paths_map_into_the_directory(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, _SandboxStub({}, workspace_dir="/workspace"), system_files_path=".octop")
        assert ws.resolve_path("SOUL.md") == "/workspace/SOUL.md"
        assert ws.resolve_path("./notes/a.md") == "/workspace/notes/a.md"
        assert ws.resolve_path(".") == "/workspace"
        assert ws.resolve_path("skills/demo") == "/workspace/.octop/skills/demo"

    def test_absolute_paths_below_the_directory_get_the_system_prefix(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, _SandboxStub({}, workspace_dir="/workspace"), system_files_path=".octop")
        assert ws.resolve_path("/workspace/skills/demo") == "/workspace/.octop/skills/demo"
        assert ws.resolve_path("/workspace/.octop/skills/demo") == "/workspace/.octop/skills/demo"
        assert ws.resolve_path("/workspace/notes/a.md") == "/workspace/notes/a.md"

    def test_absolute_paths_outside_the_directory_are_left_alone(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, _SandboxStub({}, workspace_dir="/workspace"), system_files_path=".octop")
        assert ws.resolve_path("/opt/octop/skills/demo") == "/opt/octop/skills/demo"
        assert ws.resolve_path("/tmp/out.txt") == "/tmp/out.txt"

    def test_reads_go_to_the_directory(self, tmp_path: Path) -> None:
        backend = _SandboxStub({"/workspace/SOUL.md": "soul", "/SOUL.md": "wrong"}, workspace_dir="/workspace")
        ws = _workspace(tmp_path, backend)
        assert ws.read_text("SOUL.md") == "soul"
        assert ws.exists("SOUL.md")
        assert not ws.exists("MEMORY.md")

    def test_deepagents_source_paths_use_the_directory(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, _SandboxStub({}, workspace_dir="/workspace"), system_files_path=".octop")
        assert ws.memory_paths(("AGENTS.md",), include_missing=True) == ["/workspace/AGENTS.md"]
        paths = ws.skill_paths(extra=["/opt/octop/skills"])
        assert paths[0] == "/workspace/.octop/_builtin_skills"
        assert "/workspace/.octop/skills" in paths
        assert paths[-1] == "/opt/octop/skills"

    def test_listed_paths_are_workspace_relative(self, tmp_path: Path) -> None:
        backend = _SandboxStub({"/workspace/a/b.md": "x", "/workspace/a/c.md": "y"}, workspace_dir="/workspace")
        ws = _workspace(tmp_path, backend)
        assert [row["path"] for row in ws.list_dir("a") or []] == ["a/b.md", "a/c.md"]

    def test_present_path(self, tmp_path: Path) -> None:
        ws = _workspace(tmp_path, _SandboxStub({}, workspace_dir="/workspace"))
        assert ws.present_path("/workspace/a/b.md") == "a/b.md"
        assert ws.present_path("/workspace") == "."
        # Outside the workspace the sandbox path is kept recognisable, not re-homed.
        assert ws.present_path("/tmp/out.txt") == "tmp/out.txt"
        assert ws.present_path("/workspacefoo/x") == "workspacefoo/x"
