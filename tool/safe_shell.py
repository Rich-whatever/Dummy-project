"""
SafeShellTool — A LangChain BaseTool for sandboxed shell command execution.

Uses a chroot-lite approach: the working directory is locked to ``Dummy_Folder``.
Commands that would escape are blocked.  Dangerous operations are auto-denied.
Potentially harmful operations prompt the user for approval via the terminal.

Registered in the ``filesystem_and_shell`` toolkit alongside the filesystem
MCP tools.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import SANDBOX_DIR

# ── Sandbox Root ───────────────────────────────────────────────────

# Repo-relative (``<project root>/Dummy_Folder``) — see ``config.SANDBOX_DIR``,
# so the sandbox works from any clone location.
DUMMY_FOLDER = os.path.abspath(SANDBOX_DIR)

# ── Timeout Config ─────────────────────────────────────────────────

# Ordered list of (regex_pattern, timeout_seconds). First match wins.
# Each pattern is compiled once at import time.
COMMAND_TIMEOUT_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"pip\s+install", re.IGNORECASE), 300),
    (re.compile(r"npm\s+install", re.IGNORECASE), 300),
    (re.compile(r"git\s+clone", re.IGNORECASE), 120),
    (re.compile(r"(wget|curl)\s", re.IGNORECASE), 120),
    (re.compile(r"^(python|node)\s", re.IGNORECASE), 60),
    # Generic script execution (python file.py, node script.js)
    (re.compile(r"^(python|node|ruby|perl|php)\s", re.IGNORECASE), 60),
]
DEFAULT_TIMEOUT = 30  # seconds


def _resolve_timeout(command: str) -> int:
    """Return the appropriate timeout for *command* based on pattern matching."""
    for pattern, timeout in COMMAND_TIMEOUT_PATTERNS:
        if pattern.search(command):
            return timeout
    return DEFAULT_TIMEOUT


# ── Pattern Classifications ────────────────────────────────────────

# Each entry: (compiled_regex, human_readable_reason)
# Priority order: DANGEROUS checked first, then WARN, then SAFE.

DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ── Filesystem destruction ─────────────────────────────────
    (re.compile(r"\brm\s+-rf\s+[/\\](?:\s|$)"), "Recursive deletion of root filesystem"),
    (re.compile(r"\brm\s+-rf\s+~[/\\]?"), "Recursive deletion of home directory"),
    (re.compile(r"\brm\s+-rf\s+%USERPROFILE%", re.IGNORECASE), "Recursive deletion of user profile"),
    (re.compile(r"\brm\s+-rf\s+\$HOME", re.IGNORECASE), "Recursive deletion of home directory"),
    # ── Raw disk operations ────────────────────────────────────
    (re.compile(r"\bdd\s+if="), "Raw disk write operation"),
    (re.compile(r"\bmkfs\."), "Filesystem formatting"),
    (re.compile(r"\bfdisk\b"), "Disk partition manipulation"),
    (re.compile(r"\bformat\s+[a-zA-Z]:[/\\]?"), "Drive formatting"),
    # ── Fork bombs ─────────────────────────────────────────────
    (re.compile(r":\(\)\{"), "Fork bomb detection"),
    (re.compile(r":\(\)\|\{"), "Fork bomb detection"),
    # ── System control ─────────────────────────────────────────
    (re.compile(r"\bshutdown\b"), "System shutdown"),
    (re.compile(r"\breboot\b"), "System reboot"),
    (re.compile(r"\bhalt\b"), "System halt"),
    # ── Opening entire filesystem permissions ──────────────────
    (re.compile(r"chmod\s+-R\s+777\s+[/\\]"), "Opening entire filesystem permissions"),
    # ── Device write ───────────────────────────────────────────
    (re.compile(r">\s+/dev/sd[a-z]"), "Raw device write"),
    (re.compile(r">\s+\\\\\\.\\\\[a-zA-Z]:"), "Raw Windows device write"),
]

WARN_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ── Deletion ───────────────────────────────────────────────
    (re.compile(r"\brm\b(?!\s+-rf\s+[/\\])"), "Deleting files or directories"),
    (re.compile(r"\bdel\s+", re.IGNORECASE), "Deleting files"),
    (re.compile(r"\brmdir\s+", re.IGNORECASE), "Removing directories"),
    # ── Privilege escalation ───────────────────────────────────
    (re.compile(r"\bsudo\b"), "Privilege escalation (sudo)"),
    (re.compile(r"\brunas\b"), "Privilege escalation (runas)"),
    # ── Network operations ─────────────────────────────────────
    (re.compile(r"\bwget\s+", re.IGNORECASE), "Downloading from network"),
    (re.compile(r"\bcurl\s+", re.IGNORECASE), "Network request"),
    # ── Package installation ───────────────────────────────────
    (re.compile(r"pip\s+install", re.IGNORECASE), "Installing Python packages"),
    (re.compile(r"npm\s+install", re.IGNORECASE), "Installing Node packages"),
    (re.compile(r"(apt|apt-get|yum|brew|choco|scoop)\s+install", re.IGNORECASE),
     "Installing system packages"),
    # ── Git modifications ──────────────────────────────────────
    (re.compile(r"\bgit\s+push\b"), "Pushing to git remote"),
    (re.compile(r"\bgit\s+commit\b"), "Creating git commit"),
    (re.compile(r"\bgit\s+reset\b"), "Resetting git history"),
    # ── Permission changes ─────────────────────────────────────
    (re.compile(r"\bchmod\b"), "Changing file permissions"),
    (re.compile(r"\bchown\b"), "Changing file ownership"),
    (re.compile(r"\bicacls\b", re.IGNORECASE), "Changing Windows permissions"),
    (re.compile(r"\battrib\b", re.IGNORECASE), "Changing file attributes"),
    # ── Process management ─────────────────────────────────────
    (re.compile(r"\bkill\b"), "Killing processes"),
    (re.compile(r"\btaskkill\b", re.IGNORECASE), "Killing processes"),
    (re.compile(r"\btasklist\b", re.IGNORECASE), "Listing processes"),
    # ── Docker ─────────────────────────────────────────────────
    (re.compile(r"\bdocker\b"), "Docker commands"),
    # ── Registry ───────────────────────────────────────────────
    (re.compile(r"\breg\b"), "Registry operations"),
    (re.compile(r"\bregedit\b", re.IGNORECASE), "Registry editor"),
    # ── Redirection / piping ───────────────────────────────────
    (re.compile(r"(?<!\|)\|\s+"), "Output piping (may chain commands)"),
    (re.compile(r">\s+[^|]"), "Output redirection to file"),
    (re.compile(r">>\s+"), "Output append to file"),
    # ── File operations ────────────────────────────────────────
    (re.compile(r"\bmove\s+", re.IGNORECASE), "Moving files"),
    (re.compile(r"\bcopy\s+", re.IGNORECASE), "Copying files"),
    (re.compile(r"\bxcopy\s+", re.IGNORECASE), "Bulk file copying"),
    (re.compile(r"\bren\b", re.IGNORECASE), "Renaming files"),
    (re.compile(r"\breplace\b", re.IGNORECASE), "Replacing files"),
    (re.compile(r"\bdel\b", re.IGNORECASE), "Deleting files"),
    # ── Environment / system info ──────────────────────────────
    (re.compile(r"\bset\b", re.IGNORECASE), "Environment variable operations"),
    (re.compile(r"\bexport\b"), "Environment variable operations"),
    # ── Encoding / obfuscation attempts ────────────────────────
    (re.compile(r"base64\s+--decode", re.IGNORECASE), "Base64 decode (potential obfuscation)"),
    (re.compile(r"(certutil|openssl)\s", re.IGNORECASE), "Certificate/encryption tool"),
]


def _classify_command(command: str) -> tuple[str, str]:
    """
    Classify a command into one of three levels.

    Returns
    -------
    tuple[str, str]
        ``("safe", "")``
        ``("warn", "reason")``
        ``("dangerous", "reason")``
    """
    # Check DANGEROUS first (priority order)
    for pattern, reason in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return ("dangerous", reason)

    # Check WARN
    for pattern, reason in WARN_PATTERNS:
        if pattern.search(command):
            return ("warn", reason)

    # Default: SAFE
    return ("safe", "")


# ── Sandbox Escape Check ───────────────────────────────────────────

def _check_sandbox_escape(command: str) -> str | None:
    """
    Detect attempts to change directory above the sandbox root.

    Returns an error message if escape detected, ``None`` if safe.
    """
    # Check for explicit cd attempts to parent directories
    # Match: cd .. / cd ../ / cd ../../ / cd C:\ / cd %VAR% / cd /
    escape_patterns = [
        re.compile(r"\bcd\s+\.\.[/\\]?"),          # cd ..
        re.compile(r"\bcd\s+\.\.[/\\]\.\."),        # cd ../.. (multi-level)
        re.compile(r"\bcd\s+[/\\]"),                 # cd / or cd \
        re.compile(r"\bcd\s+[a-zA-Z]:[/\\]"),        # cd C:\
        re.compile(r"\bcd\s+%[A-Z_]+%"),             # cd %VAR%
        re.compile(r"\bcd\s+\$[A-Z_]+"),             # cd $VAR
    ]

    # Extract the cd target
    for cmd_part in _split_commands(command):
        stripped = cmd_part.strip()
        for pattern in escape_patterns:
            if pattern.search(stripped):
                return (
                    "[WARNING] Sandbox escape detected.\n"
                    "You attempted to navigate above the sandbox root "
                    f"('{DUMMY_FOLDER}').\n"
                    "Stay within Dummy_Folder and its subdirectories."
                )

    return None


def _split_commands(command: str) -> list[str]:
    """Split chained commands (``&&``, ``||``, ``;``) into individual parts."""
    parts = []
    # Split on &&, ||, ; (but not inside quotes)
    for separator in ["&&", "||", ";"]:
        if separator in command:
            for part in command.split(separator):
                parts.extend(_split_commands(part))
            return parts
    return [command]


# ── User Approval ──────────────────────────────────────────────────

def _get_user_approval(command: str, reason: str) -> bool:
    """Print a warning to stderr and prompt the user for approval."""
    print(
        f"\n  [WARNING] {reason}",
        file=sys.stderr,
    )
    print(f"     Command: {command[:200]}", file=sys.stderr)
    response = input("     Approve?(just input y/n/v) [y]es / [n]o / [v]iew full: ").strip().lower()

    if response == "v":
        print(f"     Full command:\n{command}", file=sys.stderr)
        response = input("     Approve?(just input y/n) [y]es / [n]o: ").strip().lower()

    return response == "y"


# ── Input Schema ───────────────────────────────────────────────────

class ShellInput(BaseModel):
    """Input schema for SafeShellTool."""

    command: str = Field(
        description=(
            "The shell command to execute. "
            "Must be a valid cmd.exe command. "
            "Examples: 'dir', 'echo hello', 'python -c \"print(1+1)\"', "
            "'type myfile.txt'"
        )
    )


# ── Tool Implementation ────────────────────────────────────────────

class SafeShellTool(BaseTool):
    """
    LangChain BaseTool for sandboxed shell command execution.

    The working directory is locked to ``Dummy_Folder``.  The tool
    classifies commands into three categories and acts accordingly:

    * **SAFE** — executes automatically
    * **WARN** — prompts the user for approval via the terminal
    * **DANGEROUS** — denied immediately with an explanation
    """

    name: str = "safe_shell"
    description: str = f"""\
Execute shell commands in a sandboxed environment.

=== ENVIRONMENT ===
- Working directory: {DUMMY_FOLDER}
- You can navigate DOWN into subdirectories to explore files and run code.
- You CANNOT navigate above Dummy_Folder (e.g., `cd ..` will be blocked).

=== BEHAVIOR ===
- SAFE commands (file reads like `type`, `dir`, `ls`, script execution
  with `python`, `node`, etc.) run automatically.
- WARN commands (deleting files, installing packages, network operations,
  git modifications, permission changes, piping/redirection) require
  explicit user approval via a terminal prompt.
- DANGEROUS commands (filesystem destruction, disk formatting, fork bombs,
  system shutdown) are DENIED automatically with no option to override.

=== IMPORTANT RULES ===
1. If a command is denied or requires approval, DO NOT try to bypass it
   with tricks like encoding, wrapping in `python -c`, or chaining commands.
   The restriction is intentional — accept it and find another approach.
2. If you need to install a package, use `pip install`. The user will be
   prompted for approval.
3. To run Python code, use `python -c "your code here"`.
4. Always check what files exist before operating on them (use `dir` or `type`).
5. For file system navigation, you can use `cd foldername` to enter a
   subdirectory, but NOT `cd ..` to go above Dummy_Folder.

=== OUTPUT ===
Returns stdout, stderr, and the exit code of the command in a formatted block.
"""

    args_schema: Type[BaseModel] = ShellInput

    def _run(self, command: str, **kwargs: Any) -> str:
        """
        Execute *command* through the sandboxed shell.

        Steps:
        1. Classify the command
        2. Check for sandbox escape attempts
        3. If DANGEROUS -> return denial
        4. If WARN -> prompt user, return if denied
        5. Execute with timeout
        6. Check post-execution path escape
        7. Return formatted output
        """
        # ── Step 1: Classify ────────────────────────────────────────
        level, reason = _classify_command(command)

        # ── Step 2: Sandbox escape check ────────────────────────────
        escape_error = _check_sandbox_escape(command)
        if escape_error:
            return self._format_result(
                stdout="",
                stderr=escape_error,
                exit_code=1,
            )

        # ── Step 3: DANGEROUS -> auto-deny ──────────────────────────
        if level == "dangerous":
            denial = (
                f"[DENIED] {reason}\n\n"
                f"Command: {command}\n\n"
                "This operation is considered too dangerous and has been "
                "automatically blocked. Do not try to bypass this restriction."
            )
            return self._format_result(
                stdout="",
                stderr=denial,
                exit_code=1,
            )

        # ── Step 4: WARN -> prompt user ─────────────────────────────
        if level == "warn":
            approved = _get_user_approval(command, reason)
            if not approved:
                denial = (
                    f"[DENIED] Command denied by user: {reason}\n\n"
                    "Do not try to bypass this restriction with encoding, "
                    "wrapping, or alternative commands. Accept the limitation."
                )
                return self._format_result(
                    stdout="",
                    stderr=denial,
                    exit_code=1,
                )

        # ── Step 5: Execute ─────────────────────────────────────────
        timeout = _resolve_timeout(command)

        try:
            # NOTE: pass the command as a *string* with shell=True instead of
            # ["cmd.exe", "/c", command]. The list form goes through
            # subprocess.list2cmdline(), which escapes embedded double-quotes
            # as \" — cmd.exe's /c parser mangles that and echoes the command
            # text back instead of executing it (exit code 0, looks like
            # success). shell=True hands the string to cmd.exe verbatim, so
            # quoted arguments (e.g. powershell -Command "...") run correctly.
            result = subprocess.run(
                command,
                capture_output=True,
                shell=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=DUMMY_FOLDER,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            return self._format_result(
                stdout="",
                stderr=(
                    f"[TIMEOUT] Command timed out after {timeout} seconds.\n"
                    "The command was terminated.\n\n"
                    f"Command: {command[:200]}"
                ),
                exit_code=1,
            )
        except FileNotFoundError:
            return self._format_result(
                stdout="",
                stderr="cmd.exe not found. This tool requires Windows.",
                exit_code=1,
            )
        except Exception as exc:
            return self._format_result(
                stdout="",
                stderr=f"Unexpected execution error: {exc}",
                exit_code=1,
            )

        # ── Step 6: Post-execution escape check ─────────────────────
        # If the command changed directory, verify we're still in sandbox
        if "cd " in command.lower():
            try:
                check_result = subprocess.run(
                    "cd && echo __ESCAPE_CHECK__",
                    capture_output=True,
                    shell=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    cwd=DUMMY_FOLDER,
                )
                current_dir = (check_result.stdout or "").strip()
                # The output before __ESCAPE_CHECK__ is the current directory
                if "__ESCAPE_CHECK__" in current_dir:
                    current_dir = current_dir.split("__ESCAPE_CHECK__")[0].strip()
                if current_dir and not current_dir.startswith(DUMMY_FOLDER):
                    stderr += (
                        f"\n[WARNING] Post-execution: current directory "
                        f"('{current_dir}') is outside sandbox. "
                        "Changes may not persist."
                    )
            except Exception:
                pass

        # ── Step 7: Format output ───────────────────────────────────
        return self._format_result(stdout, stderr, exit_code)

    # ── Async stub ──────────────────────────────────────────────────

    async def _arun(self, command: str, **kwargs: Any) -> str:
        """Async execution not supported (sync-only tool)."""
        return self._run(command, **kwargs)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _format_result(stdout: str, stderr: str, exit_code: int) -> str:
        """Format execution result as a readable string."""
        max_output = 10_000  # character limit to prevent flooding

        # Truncate if too long
        if len(stdout) > max_output:
            stdout = stdout[:max_output] + "\n... (output truncated)"
        if len(stderr) > max_output:
            stderr = stderr[:max_output] + "\n... (output truncated)"

        parts = []
        if stdout:
            parts.append(f"[STDOUT]\n{stdout}")
        if stderr:
            parts.append(f"[STDERR]\n{stderr}")
        parts.append(f"[EXIT CODE] {exit_code}")

        return "\n\n".join(parts) if parts else "(no output)"


# ── Quick test (if run directly) ───────────────────────────────────

if __name__ == "__main__":
    tool = SafeShellTool()
    print("=== SafeShellTool Quick Test ===\n")

    tests = [
        ("dir", "SAFE: list directory"),
        ("echo hello world", "SAFE: echo"),
        ("rm test.txt", "WARN: delete file"),
        ("rm -rf /", "DANGEROUS: root deletion"),
        ("cd ..", "ESCAPE: cd parent"),
        ("pip install pandas", "WARN: pip install"),
        ("python -c \"print('Hello from python')\"", "SAFE: python -c"),
        ("sudo apt install nginx", "WARN: sudo + install"),
    ]

    import sys

    # Skip interactive WARN tests when running non-interactively
    non_interactive = not sys.stdin.isatty()

    for cmd, desc in tests:
        level, _ = _classify_command(cmd)
        if non_interactive and level == "warn":
            print(f"\n== Test: {desc} ==")
            print(f"  Command: {cmd}")
            print("  [SKIPPED: requires interactive approval]")
            continue

        print(f"\n== Test: {desc} ==")
        print(f"  Command: {cmd}")
        try:
            result = tool._run(cmd)
        except EOFError:
            print("  [SKIPPED: non-interactive environment]")
            continue
        # Print only first 5 lines of result for brevity
        lines = result.split("\n")
        for line in lines[:5]:
            print(f"  {line}")
        if len(lines) > 5:
            print(f"  ... ({len(lines) - 5} more lines)")
