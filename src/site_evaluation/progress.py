"""
Progress of the long commands, and stopping them with Ctrl-C.

On a terminal, Progress draws a bar (tqdm): how many are done, the time taken and the time left. Messages print above
it. When the output goes to a file, it prints a plain line instead, at every 5% and at least every 5 minutes.

Ctrl-C: the worker processes never see it (see worker_init), so the command decides what stops. thread_map and
on_first_ctrl_c let the first Ctrl-C start nothing new and finish what is running (so its results are saved); a
second stops at once.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _print(*x) -> None:
    print(*x, flush=True)


def _terminal() -> bool:
    """Whether a bar can be drawn: stderr is a terminal with a width."""
    try:
        return sys.stderr.isatty() and os.get_terminal_size(sys.stderr.fileno()).columns > 0
    except (OSError, ValueError):
        return False


class Progress:
    """Counts finished items (thread-safe). total None: unknown. postfix: a short status shown after the count."""

    def __init__(self, total: int | None, desc: str, unit: str = "run", log=None, every: float = 300.0):
        self.total, self.desc, self.unit, self.every = total, desc, unit, every
        self.log = log or _print
        self.n, self.t0, self.postfix = 0, time.time(), ""
        self.lock = threading.RLock()
        self._last_t, self._last_step = self.t0, 0
        self.bar = None
        if _terminal():
            from tqdm import tqdm
            self.bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, smoothing=0.05)

    def update(self, n: int = 1, postfix: str | None = None) -> None:
        with self.lock:
            self.n += n
            if postfix is not None:
                self.postfix = postfix
            if self.bar is not None:
                if postfix is not None:
                    self.bar.set_postfix_str(postfix, refresh=False)
                self.bar.update(n)
                return
            now = time.time()
            step = int(20 * self.n / self.total) if self.total else 0          # 5% steps
            if step > self._last_step or now - self._last_t >= self.every:
                self._last_t, self._last_step = now, max(step, self._last_step)
                self.log(self.line())

    def line(self) -> str:
        el = time.time() - self.t0
        out = f"[{stamp()}] {self.desc}: {self.n}"
        if self.total:
            out += f"/{self.total} {self.unit}s ({100 * self.n / self.total:.0f}%), {duration(el)}"
            if 0 < self.n < self.total:
                out += f", ~{duration(el / self.n * (self.total - self.n))} left"
        else:
            out += f" {self.unit}s, {duration(el)}"
        return out + (f" | {self.postfix}" if self.postfix else "")

    def write(self, msg: str) -> None:
        """A message, above the bar on a terminal."""
        with self.lock:
            if self.bar is not None:
                self.bar.write(msg)
            else:
                self.log(msg)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def thread_map(fn, items, jobs: int, say=None, finish_running: bool = True) -> list:
    """fn(item) for every item, jobs at a time; the results in order. An error stops it: nothing new starts and the
    error is raised. The first Ctrl-C starts nothing new and (finish_running) waits for the running items, so their
    results are saved, then raises KeyboardInterrupt; a second Ctrl-C raises at once."""
    items = list(items)
    if not items:
        return []
    ex = ThreadPoolExecutor(max(1, min(jobs, len(items))))
    futs = [ex.submit(fn, x) for x in items]
    try:
        out = [f.result() for f in futs]
    except KeyboardInterrupt:
        for f in futs:
            f.cancel()
        running = [f for f in futs if not f.done()]
        if running and finish_running:
            (say or _print)(f"Stopping: finishing the {len(running)} running (Ctrl-C again to stop now)")
            try:
                wait(running)
            except KeyboardInterrupt:
                pass
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    except BaseException:
        for f in futs:
            f.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    return out


@contextmanager
def on_first_ctrl_c(action):
    """While inside, the first Ctrl-C calls action() instead of raising; a second raises KeyboardInterrupt."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(sig, frame):
        signal.signal(signal.SIGINT, signal.default_int_handler)
        action()

    old = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)


def worker_init() -> None:
    """For worker processes. The worker leaves the terminal's process group, so Ctrl-C reaches only the main
    process, which decides what stops (ignoring SIGINT is not enough: polars catches it while it runs a query). Out of
    that group it no longer gets the terminal's hang-up either, so it exits by itself when its parent does."""
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def read_jsonl(path, repair: bool = False) -> list[dict]:
    """The rows of a JSON-lines file. An unfinished last line (being written, or cut off by a hard stop such as
    kill -9) is left out. repair (only for a file's one writer, before it appends): cut that line from the file, so
    the next row starts on a line of its own. An unreadable line elsewhere is skipped with a note."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        if repair:
            with open(path, "r+b") as f:
                f.truncate(cut)
            print(f"note: cut an unfinished last line from {path}", flush=True)
        data = data[:cut]
    rows, bad = [], 0
    for line in data.splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"note: skipped {bad} unreadable line(s) in {path}", flush=True)
    return rows
