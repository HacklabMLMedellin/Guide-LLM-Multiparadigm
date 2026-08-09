"""
launcher.py  —  Program 1
Spawns counter.py as a subprocess, then lets you:
  • watch live data from it
  • kill it at any time
  • see a summary when it stops
"""

import subprocess
import sys
import os
import time
import signal
import threading

# ── colour helpers (no external deps) ────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"


def banner():
    print(
        f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════╗
║        PROCESS LAUNCHER & MONITOR            ║
║   Watching: counter.py  (counts to 1 000)    ║
╚══════════════════════════════════════════════╝{RESET}
"""
    )


def separator():
    print(f"{DIM}{'─' * 50}{RESET}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    banner()

    # Locate counter.py next to this file
    here = os.path.dirname(os.path.abspath(__file__))
    counter_path = os.path.join(here, "counter.py")

    if not os.path.exists(counter_path):
        print(f"{RED}✗ counter.py not found at: {counter_path}{RESET}")
        sys.exit(1)

    print(f"{GREEN}▶  Launching counter.py …{RESET}")
    separator()

    # Launch the subprocess; pipe its stdout so we can read it
    proc = subprocess.Popen(
        [sys.executable, counter_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    print(f"{CYAN}PID : {BOLD}{proc.pid}{RESET}")
    print(f"{CYAN}CMD : {sys.executable} counter.py{RESET}")
    print(f"{DIM}Press  ENTER  at any time to kill the process.{RESET}\n")
    separator()

    # ── shared state ─────────────────────────────────────────────────────────
    lines_seen = []
    killed_by_us = False
    lock = threading.Lock()

    # ── thread: read stdout from child and print it ──────────────────────────
    def reader():
        for line in proc.stdout:
            stripped = line.rstrip()
            with lock:
                lines_seen.append(stripped)
            print(f"  {stripped}")
        # drain stderr (shown only if the process errored)
        err = proc.stderr.read().strip()
        if err:
            print(f"\n{RED}[stderr]{RESET}\n{err}")

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    # ── thread: wait for ENTER key to kill ───────────────────────────────────
    def killer():
        nonlocal killed_by_us
        input()  # blocks until user presses ENTER
        with lock:
            if proc.poll() is None:  # still running?
                killed_by_us = True
                proc.send_signal(signal.SIGTERM)
                print(f"\n{YELLOW}⚡ SIGTERM sent to PID {proc.pid} …{RESET}")

    killer_thread = threading.Thread(target=killer, daemon=True)
    killer_thread.start()

    # ── wait for child to finish ──────────────────────────────────────────────
    proc.wait()
    reader_thread.join(timeout=2)

    # ── summary ───────────────────────────────────────────────────────────────
    separator()
    print(f"\n{BOLD}═══════════  SESSION SUMMARY  ═══════════{RESET}")
    print(f"  PID          : {proc.pid}")
    print(f"  Return code  : {proc.returncode}")

    with lock:
        total = len(lines_seen)
        if total:
            # Try to parse the last numeric value counter.py printed
            last_line = lines_seen[-1]
            try:
                # counter.py emits lines like  "Count: 42  |  …"
                last_count = int(last_line.split("|")[0].split(":")[1].strip())
            except Exception:
                last_count = "?"

            print(f"  Lines read   : {total}")
            print(f"  Last count   : {last_count}")
            print(f"  First output : {lines_seen[0]}")
            print(f"  Last output  : {last_line}")

    if killed_by_us:
        print(f"\n  {YELLOW}● Process was killed by you (SIGTERM).{RESET}")
    elif proc.returncode == 0:
        print(f"\n  {GREEN}● Process finished naturally (reached 1 000).{RESET}")
    else:
        print(f"\n  {RED}● Process exited with code {proc.returncode}.{RESET}")

    separator()
    print(f"{DIM}Done. Goodbye.{RESET}\n")


if __name__ == "__main__":
    main()
