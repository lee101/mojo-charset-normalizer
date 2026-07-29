"""Benchmarks against charset-normalizer on identical payloads."""

from __future__ import annotations

import gc
import os
import platform
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import charset_normalizer as upstream  # noqa: E402
import mojo_charset_normalizer as mojo  # noqa: E402


def repeat_to_size(unit: str, encoding: str, size: int) -> bytes:
    encoded = unit.encode(encoding)
    return encoded * max(1, size // len(encoded))


def best_time(function, repetitions: int = 5) -> float:
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


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="ascii") as fp:
            for line in fp:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


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
    russian = repeat_to_size(
        "Это русский текст с несколькими предложениями для проверки кодировки. ",
        "cp1251",
        2_000_000,
    )
    japanese = repeat_to_size(
        "これは日本語の文章です。文字コードの検出を正しく確認します。",
        "shift_jis",
        2_000_000,
    )
    binary = bytes(range(256)) * 31_250

    cases = [
        ("ASCII detection, 8 MB", ascii_payload, False),
        ("UTF-8 detection, 8 MB", utf8_payload, False),
        ("CP1251 detection, 2 MB", russian, False),
        ("Shift-JIS detection, 2 MB", japanese, False),
        ("binary classification, 8 MB", binary, True),
    ]

    rows = []
    for name, payload, binary_case in cases:
        if binary_case:
            ours_fn = lambda p=payload: mojo.is_binary(p)
            upstream_fn = lambda p=payload: upstream.is_binary(p)
        else:
            ours_fn = lambda p=payload: mojo.from_bytes(p).best().encoding
            upstream_fn = lambda p=payload: upstream.from_bytes(p).best().encoding
        ours_result = ours_fn()
        upstream_result = upstream_fn()
        if binary_case:
            assert ours_result == upstream_result
        else:
            ours_match = mojo.from_bytes(payload).best()
            upstream_match = upstream.from_bytes(payload).best()
            assert ours_match is not None and upstream_match is not None
            assert str(ours_match) == str(upstream_match)
        ours = best_time(ours_fn)
        reference = best_time(upstream_fn)
        rows.append((name, ours, reference, reference / ours))

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.release()}")
    print(f"Python {platform.python_version()}, charset-normalizer {upstream.__version__}")
    print()
    print("| case | Mojo port | upstream | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, reference, speedup in rows:
        print(
            f"| {name} | {ours * 1000:.2f} ms | {reference * 1000:.2f} ms | "
            f"{speedup:.2f}x |"
        )


if __name__ == "__main__":
    main()
