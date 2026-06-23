"""Background GPU sampler for the FPS benchmarks.

Polls `nvidia-smi` on its own thread while the tracker runs and reports the
average and peak of:

  * process GPU memory  — memory attributed to *this* PID (via
    `--query-compute-apps`), i.e. what the method actually consumes;
  * GPU utilisation (%) — busy fraction of the device the process runs on;
  * GPU power draw (W)  — board power of that device.

The device is resolved from the PID's compute-apps row (its `gpu_uuid`), so it
works regardless of `CUDA_VISIBLE_DEVICES`. CPU-only methods (e.g. SIFT+PnP)
simply show ~0 process memory and idle utilisation, which is the honest answer.

Usage:
    with GpuMonitor() as mon:
        ... run tracker ...
    mon.report("icp")
"""

import os
import subprocess
import threading
import time


def _smi(query: str, mode: str = "gpu") -> list[list[str]]:
    """Run one nvidia-smi query, return rows of stripped string fields."""
    flag = "--query-gpu" if mode == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        if line.strip():
            rows.append([f.strip() for f in line.split(",")])
    return rows


def _to_float(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None  # "[N/A]", "[Not Supported]", etc.


class GpuMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.pid = os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.mem: list[float] = []  # process GPU memory, MiB
        self.util: list[float] = []  # device utilisation, %
        self.power: list[float] = []  # device power draw, W

    # ------------------------------------------------------------------
    def _sample(self) -> None:
        # 1) process memory + which device this PID runs on.
        proc_mem = 0.0
        my_uuid: str | None = None
        for row in _smi("pid,used_memory,gpu_uuid", mode="apps"):
            if len(row) >= 3 and row[0] == str(self.pid):
                m = _to_float(row[1])
                if m is not None:
                    proc_mem += m
                my_uuid = row[2]
        self.mem.append(proc_mem)

        # 2) utilisation + power for the device the PID uses (fallback: device 0).
        rows = _smi("uuid,utilization.gpu,power.draw", mode="gpu")
        chosen = None
        for row in rows:
            if my_uuid is not None and row and row[0] == my_uuid:
                chosen = row
                break
        if chosen is None and rows:
            chosen = rows[0]
        if chosen and len(chosen) >= 3:
            u, p = _to_float(chosen[1]), _to_float(chosen[2])
            if u is not None:
                self.util.append(u)
            if p is not None:
                self.power.append(p)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    # ------------------------------------------------------------------
    def __enter__(self) -> "GpuMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _stat(xs: list[float]) -> tuple[float, float]:
        return (sum(xs) / len(xs), max(xs)) if xs else (0.0, 0.0)

    def summary(self) -> dict:
        mem_avg, mem_peak = self._stat(self.mem)
        util_avg, util_peak = self._stat(self.util)
        pow_avg, pow_peak = self._stat(self.power)
        return {
            "samples": len(self.mem),
            "mem_avg_mib": mem_avg,
            "mem_peak_mib": mem_peak,
            "util_avg_pct": util_avg,
            "util_peak_pct": util_peak,
            "power_avg_w": pow_avg,
            "power_peak_w": pow_peak,
        }

    def report(self, label: str, torch_module=None) -> dict:
        s = self.summary()
        print(f"\n── GPU usage [{label}]  ({s['samples']} samples) ──")
        print(
            f"  process mem : avg {s['mem_avg_mib']:8.1f} MiB   "
            f"peak {s['mem_peak_mib']:8.1f} MiB"
        )
        print(
            f"  utilisation : avg {s['util_avg_pct']:8.1f} %     "
            f"peak {s['util_peak_pct']:8.1f} %"
        )
        print(
            f"  power draw  : avg {s['power_avg_w']:8.1f} W     "
            f"peak {s['power_peak_w']:8.1f} W"
        )
        if torch_module is not None and torch_module.cuda.is_available():
            # Process-exact PyTorch allocator peak — complements the nvidia-smi
            # board-level numbers above (which include the CUDA context + cuDNN).
            alloc = torch_module.cuda.max_memory_allocated() / 1024**2
            resv = torch_module.cuda.max_memory_reserved() / 1024**2
            print(
                f"  torch alloc : peak {alloc:8.1f} MiB   "
                f"reserved peak {resv:8.1f} MiB"
            )
            s["torch_alloc_peak_mib"] = alloc
            s["torch_reserved_peak_mib"] = resv
        return s
