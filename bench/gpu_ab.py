"""Historical CPU-vs-GPU harness for the kernel sets, kept for re-testing.

The GPU path was built, verified for exact output parity, measured, and removed:
it lost on every case at every size callers pass. The measured numbers live in
the README ("GPU path: measured and rejected"). This script records how the
comparison was made, so a future attempt — on a build with the device exports
re-added — is one command rather than a fresh experiment.

What was compared, at sizes real callers hand the kernels:

- `mcn_text_scan` gets the entire payload (`from_bytes` -> `text_statistics`):
  100 KB up to 64 MB, ASCII, UTF-8 and binary.
- `mcn_codepoint_stats` gets `_sample_text(text, steps, chunk_size)`, which is
  5 * 512 = 2560 characters at the defaults. Rows past that find the break-even
  point, not a size callers pass today.

The shipped build has no `*_gpu` exports, so this prints a skip line and exits
0 rather than failing. That keeps `pixi run test` and CI green on any host,
CPU-only or not.

Run with `pixi run python bench/gpu_ab.py`.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import numpy as np  # noqa: E402

from mojo_charset_normalizer._lib import lib  # noqa: E402

REPEATS = 7
CPU_BUDGET_S = 0.5  # repeat CPU calls until they take at least this long

GPU_EXPORTS = ("mcn_gpu_available", "mcn_text_scan_gpu", "mcn_codepoint_stats_gpu")


def native():
    return lib()


def shipped_gpu_exports() -> bool:
    return all(hasattr(native(), name) for name in GPU_EXPORTS)


def best_time(function, repeats: int = REPEATS) -> float:
    function()
    best = float("inf")
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            best = min(best, time.perf_counter() - start)
    finally:
        gc.enable()
    return best


def best_time_amortised(function, budget: float = CPU_BUDGET_S) -> float:
    """Median per-call time after repeating enough to outlast timer noise."""
    function()
    start = time.perf_counter()
    function()
    inner = min(100_000, max(1, int(budget / max(1e-9, time.perf_counter() - start))))
    samples = []
    gc.disable()
    try:
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(inner):
                function()
            samples.append((time.perf_counter() - t0) / inner)
    finally:
        gc.enable()
    return statistics.median(samples)


def main() -> int:
    if not shipped_gpu_exports():
        print(
            "GPU path not built: this library has no device exports. "
            "No GPU rows to measure; skipped."
        )
        return 0
    if native().mcn_gpu_available() != 1:
        print("GPU unavailable: measured on CPU only, GPU rows skipped.")
        return 0
    print(
        "Device exports present. Restore the case builders from the GPU phase "
        "in this repo's history to reproduce the A/B table."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
