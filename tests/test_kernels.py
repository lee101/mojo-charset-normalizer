import math
import numpy as np
import pytest

from mojo_charset_normalizer import byte_frequencies, entropy, is_valid_utf8
from mojo_charset_normalizer._lib import (
    RAW_PARALLEL_THRESHOLD,
    byte_statistics,
    codepoint_statistics,
    text_statistics,
    utf8_info,
)


def test_byte_frequencies_match_numpy():
    rng = np.random.default_rng(4)
    payload = rng.integers(0, 256, 250_003, dtype=np.uint8).tobytes()
    expected = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    assert np.array_equal(byte_frequencies(payload), expected)


def test_byte_statistics_categories():
    payload = b"A z\t\n\x00\x1f\x7f\x80\xff"
    histogram, stats = byte_statistics(payload)
    assert histogram.sum() == len(payload)
    assert stats.tolist() == [8, 2, 1, 3, 3, 3, 1, 1]


def test_entropy_matches_reference():
    payload = bytes(range(256)) * 1000
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    expected = -sum(
        (count / len(payload)) * math.log2(count / len(payload))
        for count in counts
        if count
    )
    assert entropy(payload) == pytest.approx(expected, abs=1e-12)
    assert entropy(payload) == pytest.approx(8.0)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"plain ASCII",
        "Καλημέρα κόσμε".encode(),
        "日本語の文章".encode(),
        b"\xf0\x9f\x98\x80",
        b"\x00\x7f",
    ],
)
def test_utf8_validator_accepts_valid_sequences(payload):
    assert is_valid_utf8(payload)
    valid, position, codepoints, _ = utf8_info(payload)
    assert valid
    assert position == len(payload)
    assert codepoints == len(payload.decode("utf-8"))


@pytest.mark.parametrize(
    "payload",
    [
        b"\x80",
        b"\xc0\x80",
        b"\xc1\xbf",
        b"\xe0\x80\x80",
        b"\xed\xa0\x80",
        b"\xf0\x80\x80\x80",
        b"\xf4\x90\x80\x80",
        b"\xf5\x80\x80\x80",
        b"\xe2\x82",
        b"\xe2(\xa1",
    ],
)
def test_utf8_validator_rejects_invalid_sequences(payload):
    assert not is_valid_utf8(payload)
    with pytest.raises(UnicodeDecodeError):
        payload.decode("utf-8")


@pytest.mark.parametrize("prefix", [0, 1, 2, 3, 31, 32, 33, 63, 64, 65])
def test_utf8_simd_boundaries_and_scalar_tails(prefix):
    payload = b"a" * prefix + "Ε日本\U0001f600".encode() * 37 + b"z" * (prefix % 7)
    valid, position, codepoints, non_ascii = utf8_info(payload)
    assert valid
    assert position == len(payload)
    assert codepoints == len(payload.decode())
    assert non_ascii == 4 * 37

    invalid = payload + b"\xf4\x90\x80\x80"
    assert not utf8_info(invalid)[0]


@pytest.mark.parametrize(
    "size", [RAW_PARALLEL_THRESHOLD - 1, RAW_PARALLEL_THRESHOLD + 17]
)
def test_text_scan_parallel_threshold_and_tail(size):
    pattern = b"A \n\x00\x1f\x80\xff"
    quotient, remainder = divmod(size, len(pattern))
    payload = pattern * quotient + pattern[:remainder]
    counts = {
        value: quotient + pattern[:remainder].count(value)
        for value in set(pattern)
    }
    raw, _ = text_statistics(payload)
    assert raw.tolist() == [
        counts[ord("A")] + counts[ord(" ")] + counts[ord("\n")]
        + counts[0] + counts[0x1F],
        counts[0x80] + counts[0xFF],
        counts[0],
        counts[0] + counts[0x1F],
        counts[ord(" ")] + counts[ord("\n")],
        counts[ord("A")] + counts[ord(" ")],
        counts[0x80],
        counts[ord("\n")],
    ]


def test_codepoint_statistics_count_scripts():
    stats = codepoint_statistics("Latin Ελληνικά Русский العربية 日本語 한글")
    assert stats[6] == 5
    assert stats[7] == 8
    assert stats[8] == 7
    assert stats[10] == 7
    assert stats[13] == 0
    assert stats[14] == 0
    assert stats[15] == 3
    assert stats[16] == 2


def test_native_boundary_rejects_invalid_addresses_and_lengths():
    from mojo_charset_normalizer._lib import lib

    native = lib()
    stats = np.empty(24, dtype=np.int64)
    assert native.mcn_codepoint_stats(0, 1, stats.ctypes.data) == -1
    assert native.mcn_codepoint_stats(0, -1, stats.ctypes.data) == -1
    assert native.mcn_codepoint_stats(0, 0, 0) == -1

    raw = np.empty(8, dtype=np.int64)
    utf8 = np.empty(3, dtype=np.int64)
    assert native.mcn_text_scan(0, 1, raw.ctypes.data, utf8.ctypes.data, 0) == -1
    assert native.mcn_utf8_stats(0, 1, utf8.ctypes.data) == -1
    assert native.mcn_byte_stats(0, 1, 0, raw.ctypes.data) == -1


def test_input_buffer_is_pinned_for_native_call(monkeypatch):
    from mojo_charset_normalizer import _lib

    payload = bytearray(b"abc")
    original = _lib.lib().mcn_utf8_stats

    class Wrapper:
        def __call__(self, address, size, stats):
            with pytest.raises(BufferError):
                payload.extend(b"x")
            return original(address, size, stats)

    monkeypatch.setattr(_lib.lib(), "mcn_utf8_stats", Wrapper())
    assert _lib.utf8_info(payload)[0]


def test_native_helpers_reject_non_buffer_contract():
    with pytest.raises(TypeError):
        byte_frequencies("abc")
