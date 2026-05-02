import time


class StageTimer:
    """Tracks elapsed time for each pipeline stage."""

    def __init__(self):
        self.records = []
        self._start = None
        self._current_stage = None

    def start(self, stage_name: str):
        self._current_stage = stage_name
        self._start = time.time()

    def stop(self):
        if self._start is None:
            return
        elapsed = time.time() - self._start
        self.records.append({"stage": self._current_stage, "seconds": round(elapsed, 3)})
        self._start = None
        self._current_stage = None
        return elapsed

    def summary(self) -> list[dict]:
        total = sum(r["seconds"] for r in self.records)
        return self.records + [{"stage": "TOTAL", "seconds": round(total, 3)}]
