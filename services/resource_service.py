import os
import threading
import time

import psutil


BYTES_PER_MB = 1024 * 1024


class ProcessResourceMonitor:
    """Measure process CPU time and peak RSS increase consistently per model."""

    def __init__(self, poll_interval=0.02):
        self.process = psutil.Process(os.getpid())
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._baseline_rss = 0
        self._peak_rss = 0
        self._start_cpu = 0.0
        self._start_wall = 0.0

    def start(self):
        cpu_times = self.process.cpu_times()
        self._start_cpu = cpu_times.user + cpu_times.system
        self._start_wall = time.perf_counter()
        self._baseline_rss = self.process.memory_info().rss
        self._peak_rss = self._baseline_rss
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_memory, daemon=True)
        self._thread.start()
        return self

    def _sample_memory(self):
        while not self._stop_event.wait(self.poll_interval):
            try:
                self._peak_rss = max(self._peak_rss, self.process.memory_info().rss)
            except psutil.Error:
                return

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(self.poll_interval * 4, 0.1))

        try:
            self._peak_rss = max(self._peak_rss, self.process.memory_info().rss)
            cpu_times = self.process.cpu_times()
            end_cpu = cpu_times.user + cpu_times.system
        except psutil.Error:
            end_cpu = self._start_cpu

        return {
            "cpu_time_seconds": max(end_cpu - self._start_cpu, 0.0),
            "peak_ram_increase_mb": max(
                self._peak_rss - self._baseline_rss,
                0,
            )
            / BYTES_PER_MB,
            "wall_time_seconds": max(time.perf_counter() - self._start_wall, 0.0),
        }
