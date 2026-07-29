"""ctypes access to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_CHARSET_NORMALIZER_LIB") or os.path.join(
    ROOT, "dist", "libmojo-charset-normalizer.so"
)
SOURCE = os.path.join(ROOT, "src", "charset_normalizer.mojo")
RAW_PARALLEL_THRESHOLD = 32 * 1024 * 1024

I = ctypes.c_int64
_SIGNATURES = {
    "mcn_byte_stats": ([I, I, I, I], I),
    "mcn_utf8_stats": ([I, I, I], I),
    "mcn_text_scan": ([I, I, I, I, I], I),
    "mcn_codepoint_stats": ([I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and (not os.path.exists(SOURCE) or os.path.getmtime(LIB) >= os.path.getmtime(SOURCE))
    ):
        return LIB
    if not os.path.exists(os.path.join(ROOT, "build", "build.sh")):
        raise BuildError(f"compiled library not found at {LIB}")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def _u8(payload: bytes | bytearray) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"expected bytes or bytearray, got {type(payload).__name__}")
    return np.frombuffer(payload, dtype=np.uint8)


def _check_status(status: int, operation: str) -> None:
    if status < 0:
        raise RuntimeError(f"native {operation} rejected its buffer arguments")


def byte_statistics(payload: bytes | bytearray) -> tuple[np.ndarray, np.ndarray]:
    data = _u8(payload)
    histogram = np.empty(256, dtype=np.uint64)
    stats = np.empty(8, dtype=np.int64)
    status = int(
        lib().mcn_byte_stats(
            data.ctypes.data, data.size, histogram.ctypes.data, stats.ctypes.data
        )
    )
    _check_status(status, "byte scan")
    return histogram, stats


def byte_frequencies(payload: bytes | bytearray) -> np.ndarray:
    """Return exact counts for all 256 byte values."""
    return byte_statistics(payload)[0]


def entropy(payload: bytes | bytearray) -> float:
    """Shannon entropy in bits per byte."""
    if not payload:
        return 0.0
    counts = byte_frequencies(payload)
    probabilities = counts[counts != 0].astype(np.float64) / len(payload)
    return -float(sum(p * math.log2(p) for p in probabilities))


def utf8_info(payload: bytes | bytearray) -> tuple[bool, int, int, int]:
    data = _u8(payload)
    stats = np.empty(3, dtype=np.int64)
    position = int(lib().mcn_utf8_stats(data.ctypes.data, data.size, stats.ctypes.data))
    _check_status(position, "UTF-8 scan")
    return position == data.size, position, int(stats[0]), int(stats[1])


def text_statistics(
    payload: bytes | bytearray,
) -> tuple[np.ndarray, tuple[bool, int, int, int]]:
    data = _u8(payload)
    raw = np.empty(8, dtype=np.int64)
    utf8 = np.empty(3, dtype=np.int64)
    scratch = np.empty(32, dtype=np.int64)
    position = int(
        lib().mcn_text_scan(
            data.ctypes.data,
            data.size,
            raw.ctypes.data,
            utf8.ctypes.data,
            scratch.ctypes.data,
        )
    )
    _check_status(position, "text scan")
    return raw, (
        position == data.size,
        position,
        int(utf8[0]),
        int(utf8[1]),
    )


def is_valid_utf8(payload: bytes | bytearray) -> bool:
    return utf8_info(payload)[0]


def codepoint_statistics(text: str) -> np.ndarray:
    encoded = text.encode("utf-32-le", "surrogatepass")
    codepoints = np.frombuffer(encoded, dtype="<u4")
    stats = np.empty(24, dtype=np.int64)
    status = int(
        lib().mcn_codepoint_stats(
            codepoints.ctypes.data, codepoints.size, stats.ctypes.data
        )
    )
    _check_status(status, "codepoint scan")
    return stats


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
