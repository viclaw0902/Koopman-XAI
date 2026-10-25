from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from core import ROOT, load_config, save_json


def wait_existing(pid):
    if os.name != "nt":
        raise RuntimeError("Adopting an existing process requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise RuntimeError(f"Cannot monitor existing process {pid}")
    exit_code = ctypes.c_ulong()
    try:
        while True:
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if exit_code.value != 259:
                return exit_code.value
            time.sleep(10)
    finally:
        kernel.CloseHandle(handle)


def execute(script, seed=None):
    arguments = [sys.executable, "-u", str(ROOT / script)]
    if seed is not None:
        arguments += ["--seed", str(seed)]
    basename = f"{Path(script).stem}_{seed if seed is not None else 'all'}"
    logs = ROOT / "Results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{basename}.stdout.log").open("a", encoding="utf-8") as stdout, (logs / f"{basename}.stderr.log").open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(arguments, cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        save_json(ROOT / "Results" / "pipeline_status.json", {"state": "running", "script": script, "seed": seed, "child_pid": process.pid, "pipeline_pid": os.getpid()})
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"{script}, seed={seed}, exit={exit_code}; inspect {logs}")


def main(adopt_pid=None):
    config = load_config()
    lock = ROOT / "Results" / "pipeline.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        if adopt_pid is not None:
            save_json(ROOT / "Results" / "pipeline_status.json", {"state": "adopting_existing_training", "seed": 42, "child_pid": adopt_pid, "pipeline_pid": os.getpid()})
            if wait_existing(adopt_pid):
                raise RuntimeError("Initial seed42 training failed; see seed42_training.stderr.log")
            if not (ROOT / "Results" / "seed_42" / "dimension_selection.json").exists():
                raise RuntimeError("Initial training exited without completing dimension selection")
        for seed in config["seeds"]:
            if not (adopt_pid is not None and seed == 42):
                execute("train.py", seed)
            execute("explain.py", seed)
            execute("summarize.py")
        save_json(ROOT / "Results" / "pipeline_status.json", {"state": "complete", "seeds": config["seeds"], "pipeline_pid": os.getpid()})
    except Exception:
        save_json(ROOT / "Results" / "pipeline_status.json", {"state": "failed", "error": traceback.format_exc(), "pipeline_pid": os.getpid()})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-pid", type=int)
    arguments = parser.parse_args()
    main(arguments.adopt_pid)
