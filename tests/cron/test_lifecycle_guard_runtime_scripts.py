"""Node entrypoints retain lifecycle checks without walking JavaScript as shell (#105758)."""

import shlex
from pathlib import Path

import pytest

from cron import lifecycle_guard


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("shebang", [
    "#!/usr/bin/env node", "#!/usr/bin/env -S node --no-warnings",
    "#!/usr/bin/node", "#!/usr/bin/env nodejs",
])
def test_node_launcher_does_not_spend_remote_reads_on_source(tmp_path, remote, shebang):
    launcher = tmp_path / "factory-hook"
    source = shebang + "\n" + "".join(f"void(/synthetic_lifecycle/path{i}/);\n" for i in range(108))
    if not remote:
        launcher.write_text(source)
    reads = []

    def read_remote(path):
        reads.append(Path(path))
        return source if Path(path) == launcher else None

    assert lifecycle_guard.scan_gateway_lifecycle(
        f"{shlex.quote(str(launcher))} identity", read_remote_script=read_remote,
    ) == (False, None)
    assert reads == ([launcher] if remote else [])


@pytest.mark.parametrize("wrapper", [False, True])
def test_literal_lifecycle_in_node_launcher_is_blocked(tmp_path, wrapper):
    launcher = tmp_path / "factory-hook"
    launcher.write_text('#!/usr/bin/env node\nrequire("node:child_process").execSync("hermes gateway restart");\n')
    command = shlex.quote(str(launcher))
    if wrapper:
        script = tmp_path / "wrapper.sh"
        script.write_text(command + "\n")
        command = f"bash {shlex.quote(str(script))}"
    assert lifecycle_guard.scan_gateway_lifecycle(command) == (True, None)


@pytest.mark.parametrize("prefix", ["bash", "source", ".", "env bash", "timeout 2 bash"])
def test_explicit_shell_overrides_node_shebang(tmp_path, prefix):
    (tmp_path / "child.sh").write_text("hermes gateway restart\n")
    launcher = tmp_path / "wrapper.js"
    launcher.write_text("#!/usr/bin/env node\n//; bash child.sh\n")
    assert lifecycle_guard.scan_gateway_lifecycle(f"{prefix} {shlex.quote(str(launcher))}") == (True, None)


def test_node_visit_does_not_hide_later_forced_shell_visit(tmp_path):
    (tmp_path / "child.sh").write_text("hermes gateway restart\n")
    (tmp_path / "cli").write_text("#!/usr/bin/env node\n//; bash child.sh\n")
    (tmp_path / "first.sh").write_text("./cli\n")
    (tmp_path / "second.sh").write_text("bash cli\n")
    assert lifecycle_guard.scan_gateway_lifecycle(
        "bash first.sh; bash second.sh", cwd=str(tmp_path),
    ) == (True, None)


@pytest.mark.parametrize("shebang", ["#!/bin/sh", "#!/usr/bin/env -S -u node sh", ""])
def test_unknown_or_shell_source_retains_recursion(tmp_path, shebang):
    (tmp_path / "child.sh").write_text("hermes gateway restart\n")
    launcher = tmp_path / "wrapper.js"
    launcher.write_text(shebang + "\nbash child.sh\n")
    assert lifecycle_guard.scan_gateway_lifecycle(str(launcher)) == (True, None)


@pytest.mark.parametrize("limit", ["_MAX_LIFECYCLE_SCAN_BYTES", "_MAX_LIFECYCLE_SCAN_LINE_BYTES"])
def test_node_source_keeps_scan_budget(tmp_path, monkeypatch, limit):
    launcher = tmp_path / "cli"
    launcher.write_text("#!/usr/bin/env node\n" + "x" * 512 + "\n")
    monkeypatch.setattr(lifecycle_guard, limit, 256)
    blocked, refusal = lifecycle_guard.scan_gateway_lifecycle(str(launcher))
    assert blocked
    assert refusal
