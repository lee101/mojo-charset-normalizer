"""A/B two builds of the native kernels on identical calls.

Usage:
    python bench/ab.py OLD.so NEW.so [--mode throughput|end_to_end]

Each side re-executes this script as a child with
`MOJO_CHARSET_NORMALIZER_LIB` pointing at its library, so nothing is compared
across processes. Every call is repeated in a fresh child; interleaved A/B
rounds mean a transient spike on this shared host can only distort a round, not
a side.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Throughput kernel sizes: each call lasts long enough (>=1 ms) that the C-ABI
# entry cost is noise. A fixed 100 KB sample takes ~10 us, where call overhead
# swamps any SIMD delta.
SIZES = {"text_scan": 8_000_000, "codepoints": 400_000}
REPEATS = 9
ROUNDS = 5


def payloads() -> dict[str, bytes]:
    def repeat(unit: str, encoding: str, size: int) -> bytes:
        encoded = unit.encode(encoding)
        return encoded * max(1, size // len(encoded))

    ascii_unit = "A practical English document with ordinary words, punctuation, and 12345.\n"
    utf8_unit = "Unicode text: Ελληνικά, Русский, العربية, 日本語, français.\n"
    return {
        "ascii": repeat(ascii_unit, "ascii", SIZES["text_scan"]),
        "utf8": repeat(utf8_unit, "utf-8", SIZES["text_scan"]),
        "binary": bytes(range(256)) * (SIZES["text_scan"] // 256),
        "cp1251_text": "Это русский текст с несколькими предложениями для проверки кодировки. "
        * 3500,
        "sjis_text": "これは日本語の文章です。文字コードの検出を正しく確認します。" * 9000,
        "utf8_text": "Αυτό είναι ένα ελληνικό κείμενο για τον έλεγχο της κωδικοποίησης. "
        * 5000,
    }


def _digest(arrays) -> int:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(memoryview(a).cast("B"))
    return int.from_bytes(h.digest(), "little")


def native_cases() -> list[tuple[str, object, tuple]]:
    """(name, zero-arg callable, arrays the digest covers)."""
    import numpy as np  # noqa: E402

    from mojo_charset_normalizer._lib import lib  # noqa: E402

    native = lib()
    data = payloads()
    cases = []

    for name in ("ascii", "utf8", "binary"):
        buf = np.frombuffer(data[name], dtype=np.uint8)
        raw = np.empty(8, dtype=np.int64)
        utf8 = np.empty(3, dtype=np.int64)
        scratch = np.empty(32, dtype=np.int64)

        def run(buf=buf, raw=raw, utf8=utf8, scratch=scratch):
            native.mcn_text_scan(
                buf.ctypes.data, buf.size, raw.ctypes.data, utf8.ctypes.data,
                scratch.ctypes.data,
            )

        cases.append((f"mcn_text_scan {name}", run, (raw, utf8)))

    # Above 30 MB, where the rejected parallel raw scan was measured.
    for name in ("ascii", "utf8"):
        unit = data[name]
        count = (40 << 20) // len(unit) + 1
        big = np.frombuffer(unit * count, dtype=np.uint8)[: 40 << 20]
        raw = np.empty(8, dtype=np.int64)
        utf8 = np.empty(3, dtype=np.int64)
        scratch = np.empty(1, dtype=np.int64)

        def big_run(big=big, raw=raw, utf8=utf8, scratch=scratch):
            native.mcn_text_scan(
                big.ctypes.data, big.size, raw.ctypes.data, utf8.ctypes.data,
                scratch.ctypes.data,
            )

        cases.append((f"mcn_text_scan {name}", big_run, (raw, utf8)))

    buf = np.frombuffer(data["utf8"], dtype=np.uint8)
    stats = np.empty(3, dtype=np.int64)

    def utf8_run(buf=buf, stats=stats):
        native.mcn_utf8_stats(buf.ctypes.data, buf.size, stats.ctypes.data)

    cases.append(("mcn_utf8_stats utf8", utf8_run, (stats,)))

    for name in ("cp1251_text", "sjis_text", "utf8_text"):
        encoded = data[name].encode("utf-32-le", "surrogatepass")
        buf = np.frombuffer(encoded, dtype="<u4")
        out = np.empty(24, dtype=np.int64)

        def cp_run(buf=buf, out=out):
            native.mcn_codepoint_stats(buf.ctypes.data, buf.size, out.ctypes.data)

        cases.append((f"mcn_codepoint_stats {name}", cp_run, (out,)))

    return cases


def end_to_end_cases() -> list[tuple[str, object, tuple]]:
    import numpy as np  # noqa: E402

    from mojo_charset_normalizer import from_bytes  # noqa: E402

    data = payloads()
    cases = []
    for name in ("ascii", "utf8", "binary"):
        payload = data[name]
        out = np.zeros(1, dtype=np.int64)

        def run(payload=payload, out=out):
            match = from_bytes(payload).best()
            out[0] = -1 if match is None else len(str(match))

        cases.append((f"from_bytes {name}", run, (out,)))
    return cases


def measure(cases) -> dict[str, dict[str, float]]:
    results = {}
    for name, run, outputs in cases:
        run()
        digests = set()
        times = []
        gc.disable()
        try:
            for _ in range(REPEATS):
                start = time.perf_counter()
                run()
                times.append(time.perf_counter() - start)
                digests.add(_digest(outputs))
        finally:
            gc.enable()
        results[name] = {
            "min": min(times),
            "digest": digests.pop() if len(digests) == 1 else -1,
        }
    return results


def child(lib_path: str, mode: str) -> dict:
    env = dict(os.environ)
    env["MOJO_CHARSET_NORMALIZER_LIB"] = lib_path
    env["PYTHONPATH"] = os.path.join(ROOT, "python")
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--_child", mode],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=3600,
    )
    if out.returncode:
        raise SystemExit(f"child failed for {lib_path}:\n{out.stderr[-4000:]}")
    return json.loads(out.stdout)


def main() -> None:
    args = sys.argv[1:]
    mode = "throughput"
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    if args[:1] == ["--_child"]:
        cases = native_cases() if mode == "throughput" else end_to_end_cases()
        json.dump(measure(cases), sys.stdout)
        return

    old, new = args[0], args[1]
    samples: dict[str, dict[str, list]] = {"old": {}, "new": {}}
    for _ in range(ROUNDS):
        for side, path in (("old", old), ("new", new)):
            for name, row in child(path, mode).items():
                samples[side].setdefault(name, []).append(row)

    print(f"| case | old min | new min | per-round min delta | outputs match |")
    print("| --- | ---: | ---: | --- | --- |")
    for name in samples["new"]:
        old_rows = samples["old"][name]
        new_rows = samples["new"][name]
        old_best, new_best = min(r["min"] for r in old_rows), min(r["min"] for r in new_rows)
        per_round = [
            (n["min"] / o["min"] - 1.0) * 100
            for o, n in zip(old_rows, new_rows)
        ]
        match = "yes" if all(o["digest"] == n["digest"] for o, n in zip(old_rows, new_rows)) else "NO"
        spread = ", ".join(f"{d:+.0f}%" for d in per_round)
        print(
            f"| {name} | {old_best * 1000:.3f} ms | {new_best * 1000:.3f} ms | "
            f"{(new_best / old_best - 1.0) * 100:+.1f}% ({spread}) | {match} |"
        )
