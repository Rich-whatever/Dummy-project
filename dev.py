"""
dev.py — one-command launcher for the Dummy agent (backend + frontend).

Starts BOTH processes in one terminal:

    [backend]   python -u -m api.server     → FastAPI on 127.0.0.1:8000
    [frontend]  npm run dev                 → Vite on http://localhost:5173

and then WATCHES backend ``.py`` files. When one changes, it restarts ONLY the
backend process — the frontend (and this launcher) stay up, so you never have
to touch a second terminal again. The frontend keeps its own Vite HMR.

Usage::

    python dev.py                    # both; backend auto-restarts on .py change
    python dev.py --no-reload        # both; no file watching
    python dev.py --backend-only     # just the API
    python dev.py --frontend-only    # just Vite
    python dev.py --open             # open the UI in the browser when ready
    python dev.py --frontend "pnpm dev"   # override the frontend command

Ctrl+C stops both (the whole process tree is killed).
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
FRONTEND_URL = "http://localhost:5173"
HEALTH_URL = f"{BACKEND_URL}/api/health"

WATCH_POLL_SECONDS = 1.0
WATCH_DEBOUNCE_SECONDS = 0.6
HEALTH_TIMEOUT_SECONDS = 180.0

# Directories the backend watcher never descends into (keeps the mtime scan fast).
SKIP_DIR_NAMES = {
    ".git", ".vscode", ".clinerules", ".benchmarks", ".playwright-mcp",
    ".pytest_cache", "__pycache__", "node_modules", "chroma_db",
    "preference_chroma_db", "vector_test_db", "uploads", "Dummy_Folder",
    "frontend",
}

_USE_COLOR = sys.stdout.isatty()
_COLORS = {
    "backend": "\x1b[36m",
    "frontend": "\x1b[35m",
    "dev": "\x1b[32m",
    "warn": "\x1b[33m",
}
_RESET = "\x1b[0m"


def _log(tag: str, message: str) -> None:
    """Print one prefixed line (colorized when attached to a terminal)."""
    prefix = f"[{tag}]".ljust(11)
    color = _COLORS.get(tag, "")
    if _USE_COLOR and color:
        print(f"{color}{prefix}{_RESET} {message}", flush=True)
    else:
        print(f"{prefix} {message}", flush=True)


def _raise_keyboard_interrupt(signum, frame) -> None:  # noqa: ANN001
    """Route a console signal into the same cleanup path as Ctrl+C."""
    raise KeyboardInterrupt


# ─────────────────────────────────────────────────────────────────────
# Process helpers
# ─────────────────────────────────────────────────────────────────────


def _popen(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.Popen:
    """Start a child process whose output we stream line-by-line."""
    kwargs: dict = dict(
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if os.name == "nt":
        # Own process group so terminal Ctrl+C does not race our shutdown code.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid
    return subprocess.Popen(cmd, env=env, **kwargs)


def _stream(proc: subprocess.Popen, tag: str) -> None:
    """Reader thread: forward the child's output with a tag prefix."""
    if proc.stdout is None:
        return
    try:
        for line in proc.stdout:
            _log(tag, line.rstrip("\r\n"))
    except Exception:
        pass


def _kill_tree(proc: subprocess.Popen | None) -> None:
    """Kill a child and everything it spawned (npm→node, uvicorn→npx MCP)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _start_backend() -> subprocess.Popen:
    _log("dev", "starting backend: python -u -m api.server")
    proc = _popen([sys.executable, "-u", "-m", "api.server"], cwd=PROJECT_ROOT)
    threading.Thread(target=_stream, args=(proc, "backend"), daemon=True).start()
    return proc


def _frontend_shell_command(override: str | None) -> list[str]:
    """Build the shell-wrapped frontend command (npm.cmd needs cmd.exe)."""
    cmd = override or "npm run dev"
    if os.name == "nt":
        return ["cmd", "/c", cmd]
    return ["sh", "-c", cmd]


def _start_frontend(override: str | None) -> subprocess.Popen:
    if not override and shutil.which("npm") is None and shutil.which("npm.cmd") is None:
        _log("warn", "npm was not found on PATH -- cannot start the frontend.")
        _log("warn", "Install Node.js, or pass e.g. --frontend \"pnpm dev\".")
        raise SystemExit(1)

    cmd = _frontend_shell_command(override)
    _log("dev", f"starting frontend: {(override or 'npm run dev')}  (cwd={FRONTEND_DIR.name})")
    proc = _popen(cmd, cwd=FRONTEND_DIR)
    threading.Thread(target=_stream, args=(proc, "frontend"), daemon=True).start()
    return proc


# ─────────────────────────────────────────────────────────────────────
# Health check + file watching
# ─────────────────────────────────────────────────────────────────────


def _wait_for_health(proc: subprocess.Popen) -> bool:
    """Poll /api/health until the backend answers (or the process dies)."""
    deadline = time.time() + HEALTH_TIMEOUT_SECONDS
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _announce_ready(proc: subprocess.Popen, show_frontend: bool) -> None:
    """Background thread: print a ready banner once the API responds."""
    if _wait_for_health(proc):
        _log("dev", f"backend ready  -> {BACKEND_URL}")
        if show_frontend:
            _log("dev", f"UI ready       -> {FRONTEND_URL}")
    else:
        _log("warn", "backend did not become ready -- check the [backend] log above")


def _watch_snapshot() -> dict[str, float]:
    """Map every backend .py file (relative path) to its mtime."""
    snap: dict[str, float] = {}
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        for name in files:
            if not name.endswith(".py") or name == Path(__file__).name:
                continue
            path = Path(root) / name
            try:
                snap[str(path.relative_to(PROJECT_ROOT))] = path.stat().st_mtime
            except OSError:
                pass
    return snap


def _changed_files(before: dict, after: dict) -> list[str]:
    """Files whose mtime changed or that were added/removed."""
    changed = {k for k, v in after.items() if before.get(k) != v}
    changed |= {k for k in before if k not in after}
    return sorted(changed)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the backend + frontend together, restarting the "
                    "backend when a .py file changes.",
    )
    parser.add_argument("--no-reload", action="store_true",
                        help="Do not watch backend files.")
    parser.add_argument("--backend-only", action="store_true",
                        help="Start only the FastAPI backend.")
    parser.add_argument("--frontend-only", action="store_true",
                        help="Start only the Vite frontend.")
    parser.add_argument("--open", action="store_true",
                        help="Open the UI in the browser once it is ready.")
    parser.add_argument("--frontend", default=None, metavar="CMD",
                        help="Override the frontend command (default: npm run dev).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.backend_only and args.frontend_only:
        raise SystemExit("--backend-only and --frontend-only are mutually exclusive")

    # Ctrl+Break should clean up the same way Ctrl+C does.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _raise_keyboard_interrupt)

    _log("dev", f"project root: {PROJECT_ROOT}")

    backend: subprocess.Popen | None = None
    frontend: subprocess.Popen | None = None

    def _shutdown() -> None:
        _kill_tree(frontend)
        _kill_tree(backend)

    atexit.register(_shutdown)

    try:
        if not args.frontend_only:
            backend = _start_backend()
            threading.Thread(
                target=_announce_ready,
                args=(backend, not args.backend_only),
                daemon=True,
            ).start()

        if not args.backend_only:
            frontend = _start_frontend(args.frontend)
            if args.open:
                threading.Thread(
                    target=lambda: (time.sleep(4), webbrowser.open(FRONTEND_URL)),
                    daemon=True,
                ).start()

        if backend is None or args.no_reload:
            # No watching: block on whichever process(es) we started.
            watch_target = frontend if frontend is not None else backend
            assert watch_target is not None
            watch_target.wait()
            return

        _log("dev", "watching backend *.py for changes  (Ctrl+C to stop)")
        last = _watch_snapshot()
        backend_reported = False

        while True:
            time.sleep(WATCH_POLL_SECONDS)

            if backend.poll() is not None:
                if not backend_reported:
                    _log("warn", f"backend exited (code {backend.returncode}) -- "
                                 "edit a .py file to restart it")
                    backend_reported = True
                continue
            backend_reported = False

            snap = _watch_snapshot()
            changed = _changed_files(last, snap)
            if not changed:
                continue

            # Debounce: editors often touch several files in quick succession.
            time.sleep(WATCH_DEBOUNCE_SECONDS)
            snap = _watch_snapshot()
            changed = sorted(set(changed) | set(_changed_files(last, snap)))
            last = snap

            _log("dev", f"detected {len(changed)} changed file(s) -> restarting backend")
            for name in changed[:5]:
                _log("dev", f"   - {name}")
            if len(changed) > 5:
                _log("dev", f"   ... and {len(changed) - 5} more")

            _kill_tree(backend)
            backend = _start_backend()
            threading.Thread(
                target=_announce_ready,
                args=(backend, not args.backend_only),
                daemon=True,
            ).start()

    except KeyboardInterrupt:
        _log("dev", "stopping...")
    finally:
        _shutdown()
        _log("dev", "stopped.")


if __name__ == "__main__":
    main()

