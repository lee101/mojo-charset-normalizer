"""Micro-benchmarks for the native kernels at real caller sizes.

The end-to-end numbers in `bench/bench.py` also include Python-side work, so a
kernel change can hide behind it. This script times each export directly on the
payloads and sample sizes the Python layer actually hands it, which is what the
SIMD work is tuned against. Run it with `pixi run python bench/kernels.py`.
"""

from __future__ import annotations

import gc
import os
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import numpy as np  # noqa: E402

from mojo_charset_normalizer._lib import (  # noqa: E402
    codepoint_statistics,
    lib,
    text_statistics,
    utf8_info,
)


def repeat_to_size(unit: str, encoding: str, size: int) -> bytes:
    encoded = unit.encode(encoding)
    return encoded * max(1, size // len(encoded))


def best_time(function, repetitions: int = 7) -> float:
    function()
    result = float("inf")
    gc.disable()
    try:
        for _ in range(repetitions):
            start = time.perf_counter()
            function()
            result = min(result, time.perf_counter() - start)
    finally:
        gc.enable()
    return result


def text_scan_case(name: str, payload: bytes):
    data = np.frombuffer(payload, dtype=np.uint8)
    raw = np.empty(8, dtype=np.int64)
    utf8 = np.empty(3, dtype=np.int64)
    scratch = np.empty(32, dtype=np.int64)
    native = lib().mcn_text_scan

    def run():
        native(
            data.ctypes.data,
            data.size,
            raw.ctypes.data,
            utf8.ctypes.data,
            scratch.ctypes.data,
        )

    return name, run


def utf8_case(name: str, payload: bytes):
    data = np.frombuffer(payload, dtype=np.uint8)
    stats = np.empty(3, dtype=np.int64)
    native = lib().mcn_utf8_stats

    def run():
        native(data.ctypes.data, data.size, stats.ctypes.data)

    return name, run


def codepoint_case(name: str, text: str):
    def run():
        codepoint_statistics(text)

    return name, run


def main() -> None:
    ascii_payload = repeat_to_size(
        "A practical English document with ordinary words, punctuation, and 12345.\n",
        "ascii",
        8_000_000,
    )
    utf8_payload = repeat_to_size(
        "Unicode text: Ελληνικά, Русский, العربية, 日本語, français.\n",
        "utf-8",
        8_000_000,
    )
    binary = bytes(range(256)) * 31_250
    russian_text = (
        "Это русский текст с несколькими предложениями для проверки кодировки. "
        * 100
    )[:10_000]
    japanese_text = "これは日本語の文章です。文字コードの検出を正しく確認します。" * 200

    scans = [
        text_scan_case("mcn_text_scan ASCII 8 MB", ascii_payload),
        text_scan_case("mcn_text_scan UTF-8 8 MB", utf8_payload),
        text_scan_case("mcn_text_scan binary 8 MB", binary),
        utf8_case("mcn_utf8_stats UTF-8 8 MB", utf8_payload),
        codepoint_case("mcn_codepoint_stats CP1251 text", russian_text),
        codepoint_case("mcn_codepoint_stats Shift-JIS text", japanese_text),
    ]

    for name, run in scans:
        elapsed = best_time(run)
        print(f"{name:40s} {elapsed * 1000:8.3f} ms")


if __name__ == "__main__":
    main()
