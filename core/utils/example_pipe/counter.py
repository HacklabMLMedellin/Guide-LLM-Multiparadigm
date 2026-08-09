"""
counter.py  —  Program 2
Counts from 1 to 1 000, printing a status line every number.
Sleeps 0.5 s between counts so the process runs long enough
to be observed and killed.

Handles SIGTERM gracefully — prints a farewell before exiting.
"""

import time
import signal
import sys

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
DIM = "\033[2m"

TARGET = 1000


# ── graceful shutdown on SIGTERM ──────────────────────────────────────────────
def handle_sigterm(signum, frame):
    print(f"\n{RED}✗ SIGTERM received — shutting down gracefully.{RESET}", flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_sigterm)


# ── progress bar helper ───────────────────────────────────────────────────────
def bar(n, width=20):
    filled = int(width * n / TARGET)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


# ── milestone messages ────────────────────────────────────────────────────────
MILESTONES = {
    1: "🚀 Starting up!",
    100: "10 % done — still going…",
    250: "Quarter-way there!",
    500: "🎯 Halfway — 500!",
    750: "75 % — almost there!",
    999: "One more to go…",
    1000: "🎉 Done! Reached 1 000!",
}

# ── main loop ─────────────────────────────────────────────────────────────────
start_time = time.time()

for n in range(1, TARGET + 1):
    elapsed = time.time() - start_time
    pct = n / TARGET * 100
    eta = (elapsed / n) * (TARGET - n) if n > 1 else 0

    note = MILESTONES.get(n, "")
    note_str = f"  ← {YELLOW}{note}{RESET}" if note else ""

    line = (
        f"{CYAN}Count: {BOLD}{n:>4}{RESET}"
        f"  {bar(n)}"
        f"  {GREEN}{pct:5.1f}%{RESET}"
        f"  elapsed {elapsed:6.1f}s"
        f"  ETA {eta:6.1f}s"
        f"{note_str}"
    )

    print(line, flush=True)

    if n < TARGET:
        time.sleep(0.5)  # 0.5 s per tick → ~8.3 minutes total if not killed

print(
    f"\n{BOLD}{GREEN}Counter finished. Total time: {time.time()-start_time:.1f}s{RESET}",
    flush=True,
)
