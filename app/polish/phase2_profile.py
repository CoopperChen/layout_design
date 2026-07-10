"""Optional timing breakdown for phase-2 polish (per round)."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

_active: "Phase2Profile | None" = None


class Phase2Profile:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.round_totals: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._current_round: int | None = None

    def set_round(self, round_idx: int) -> None:
        self._current_round = round_idx

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] += seconds
        self.counts[name] += 1
        if self._current_round is not None:
            self.round_totals[self._current_round][name] += seconds

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        import time

        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def print_round_summary(self, round_idx: int) -> None:
        rt = dict(self.round_totals.get(round_idx, {}))
        if not rt:
            return
        total = sum(rt.values())
        print(f"\n--- Phase 2 profile round {round_idx + 1} ({total:.2f}s) ---")
        for name, sec in sorted(rt.items(), key=lambda item: -item[1]):
            pct = 100.0 * sec / total if total > 0 else 0.0
            print(f"  {name}: {sec:.3f}s ({pct:.0f}%)")

    def print_total_summary(self) -> None:
        if not self.totals:
            return
        total = sum(self.totals.values())
        print(f"\n=== Phase 2 profile total ({total:.2f}s) ===")
        for name, sec in sorted(self.totals.items(), key=lambda item: -item[1]):
            pct = 100.0 * sec / total if total > 0 else 0.0
            n = self.counts[name]
            per = sec / n if n else 0.0
            print(
                f"  {name}: {sec:.3f}s ({pct:.0f}%, n={n}, {per * 1000:.1f}ms/call)"
            )


def start_phase2_profile() -> Phase2Profile:
    global _active
    _active = Phase2Profile()
    return _active


def get_phase2_profile() -> Phase2Profile | None:
    return _active


def stop_phase2_profile() -> None:
    global _active
    if _active is not None:
        _active.print_total_summary()
    _active = None


@contextmanager
def profile_step(name: str) -> Iterator[None]:
    prof = get_phase2_profile()
    if prof is None:
        yield
        return
    with prof.step(name):
        yield
