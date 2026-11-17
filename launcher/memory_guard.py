"""Watch the Windows commit budget and WSL RAM during a launcher session."""

import base64
import os
from pathlib import Path
import select
import signal
import subprocess
from threading import Event, Thread
from time import monotonic

from ram_plan import GIB, RESERVE, read_meminfo


class MemoryGuard:
    def __init__(self, ssh_target, read_wsl, write_status):
        script = Path(__file__).with_name("windows-memory.ps1").read_text(encoding="utf-8")
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        self.sampler = subprocess.Popen(
            [*ssh_target, "powershell -NoProfile -EncodedCommand " + encoded],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.stop = Event()
        self.read_wsl = read_wsl
        self.write_status = write_status
        self.active = False
        self.windows = None
        self.wsl_ready = False
        self.threads = [Thread(target=target, daemon=True) for target in (self.watch_windows, self.watch_wsl)]
        for thread in self.threads:
            thread.start()

    def fail(self, message):
        if not self.stop.is_set():
            self.stop.set()
            self.write_status(message)
            os.kill(os.getpid(), signal.SIGTERM)

    def watch_windows(self):
        pending = b""
        last = monotonic()
        low_since = None
        try:
            while not self.stop.is_set():
                readable, _, _ = select.select([self.sampler.stdout], [], [], 0.5)
                if not readable:
                    if monotonic() - last > 15:
                        raise TimeoutError("Windows sampler did not return a memory reading")
                    continue
                new = os.read(self.sampler.stdout.fileno(), 4096)
                if not new:
                    raise EOFError("Windows sampler exited")
                lines = (pending + new).split(b"\n")
                pending = lines.pop()
                for line in lines:
                    available_mib, used, limit = map(int, line.split(b","))
                    if min(available_mib, used, limit) < 0 or limit == 0:
                        raise ValueError("Invalid Windows memory counters")
                    last = monotonic()
                    headroom = limit - used
                    self.windows = available_mib, headroom
                    low_since = (low_since if low_since is not None else last) if headroom < GIB else None
                    floor = 16 if self.active else 12
                    grace = 2 if self.active else 15
                    if (available_mib < floor * 1024 or headroom < 128 * 2**20
                            or (low_since is not None and last - low_since >= grace)):
                        self.fail(f"Windowsのメモリ不足で停止しました。空き {available_mib/1024:.2f} GiB、コミット余裕 {headroom/GIB:.2f} GiB。")
                        return
        except (OSError, ValueError, EOFError):
            self.fail("Windowsのメモリ監視が途切れたため停止しました。")

    def watch_wsl(self):
        try:
            while not self.stop.is_set():
                free = read_meminfo(self.read_wsl())["MemAvailable"]
                if free < RESERVE:
                    self.fail(f"WSL2の空きRAMが12 GiBを下回ったため停止しました（{free/GIB:.2f} GiB）。")
                    return
                self.wsl_ready = True
                self.stop.wait(3)
        except (OSError, ValueError, subprocess.SubprocessError):
            self.fail("WSL2のメモリ監視が途切れたため停止しました。")

    def ready(self):
        sample = self.windows
        return self.wsl_ready and sample is not None and sample[0] >= 16 * 1024 and sample[1] >= 3 * GIB

    def close(self):
        self.stop.set()
        if self.sampler.poll() is None:
            self.sampler.terminate()
        try:
            self.sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.sampler.kill()
            self.sampler.wait()
