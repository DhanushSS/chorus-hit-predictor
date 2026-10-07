"""POSIX process-group supervision for bounded training and diagnostics."""
import os
import signal
import subprocess


def require_supported_platform(platform=None):
    if (platform or os.name) != 'posix':
        raise ValueError('Supervised training/diagnostics require macOS or Linux (POSIX). Saved-run prediction and app serving do not use this supervisor.')


def stop_process_group(child):
    # Send even if the group leader exited: it may have left workers behind.
    try: os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: child.wait(timeout=3)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    child.wait()


def supervise(command, seconds, cwd):
    require_supported_platform()
    child = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return child.wait(timeout=seconds)
    finally:
        stop_process_group(child)
