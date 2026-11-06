"""Conservative interpreter classification for the lifecycle guard's shell walk."""

import shlex
from pathlib import Path
from typing import Optional


def referenced_script_candidates(
    command: str, walk_command: str, *, cwd: Optional[str], executed: bool,
) -> list[tuple[Path, bool, bool]]:
    """Executed references precede inert mentions so data cannot starve executable scans."""
    from cron.lifecycle_guard import _iter_referenced_shell_scripts

    candidates = [(path, executed, forced_shell)
                  for path, forced_shell in _iter_referenced_shell_scripts(walk_command, cwd=cwd)]
    if walk_command != command:
        candidates += [(path, False, forced_shell)
                       for path, forced_shell in _iter_referenced_shell_scripts(command, cwd=cwd)]
    return candidates


def uses_node_interpreter(text: str) -> bool:
    """Only a recognized Node shebang proves that a bare entrypoint is JavaScript.

    Unknown interpreters and env options retain shell inspection. In particular,
    an option value named ``node`` must never identify the interpreter.
    """
    if not text.startswith("#!"):
        return False
    first_line = text.partition("\n")[0]
    if len(first_line) > 4096:
        return False
    try:
        argv = shlex.split(first_line[2:])
    except ValueError:
        return False
    if not argv:
        return False
    if argv[0].rsplit("/", 1)[-1] == "env":
        argv = argv[1:]
        if argv and argv[0] == "-S":
            argv = argv[1:]
    return bool(argv) and argv[0].rsplit("/", 1)[-1] in {"node", "nodejs"}
