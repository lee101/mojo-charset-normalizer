"""Arithmetic intensity of the native kernels, in flops per input byte.

The GPU gate in `prompts/accel.md` admits only kernels above roughly 2 flops
per byte: below that, the work is memory-bound and a device copy costs more than
the arithmetic it feeds. `perf` cannot answer this on the shared host
(`perf_event_paranoid=4` blocks counters), so this counts the vector operations
in the hot loop of each kernel and divides by the bytes one iteration consumes.

Counted per SIMD vector: every `ge`/`lt`/`ne`/`eq` is one compare, every
`+`/`-`/`&`/`|` is one op, and a `cast` is free (it widens lanes). Integer
adds to per-lane accumulators are counted because they are the accumulation
that would have to move to the device. A count is an estimate of issued vector
ops, not of retired ALU instructions; the order of magnitude is the decision.

Run with `pixi run python bench/intensity.py`.
"""

from __future__ import annotations

# Measured on the pinned compiler: `simd_width_of[DType.uint8]()` is 32.
# Re-check with a one-line `mojo run` if the pin moves, because this changes
# every row below.
LANES = 32


def raw_stats_ops_per_vector() -> int:
    """`raw_stats`: 8 byte-class accumulators.

    Per vector (src/charset_normalizer.mojo):
      acc0  values.lt(c128)                                 1 cmp + 1 add
      acc1  values.ge(c128)                                 1 cmp + 1 add
      acc2  values.eq(zero)                                 1 cmp + 1 add
      acc3  ((lt32 & ne9 & ne10 & ne12 & ne13) | eq127)     6 cmp + 4 and/or + 1 add
      acc4  (eq9 | eq10 | eq12 | eq13 | eq32)               5 cmp + 4 or  + 1 add
      acc5  (ge32 & lt127)                                  2 cmp + 1 and + 1 add
      acc6  (ge128 & lt160)                                 2 cmp + 1 and + 1 add
      acc7  eq10                                            1 cmp + 1 add
    """
    return (1 + 1) + (1 + 1) + (1 + 1) + (6 + 4 + 1) + (5 + 4 + 1) + (2 + 1 + 1) + (
        2 + 1 + 1
    ) + (1 + 1)


def utf8_count_ops_per_vector() -> int:
    """`utf8_count`: validator plus two counters, over 4 shifted vectors.

    Per vector:
      continuation  current.ge(c80) & current.lt(cc2)            2 cmp + 1 and
      required      prev.ge(cc2) & prev.lt(cf5) | (two more)     6 cmp + 4 and/or
      valid_kind    current.lt(c80) | continuation |
                    (current.ge(cc2) & current.lt(cf5))          3 cmp + 3 or
      bad_special   four (eq & cmp) pairs                        8 cmp + 4 and + 3 or
      bad           cont.ne(req) | valid_kind.eq(false) | special 2 cmp + 2 or
                    + reduce_or over 16 lanes                   ~15
      high_count    current.ge(c80).select(1,0).reduce_add       1 cmp + 2 select
                    + reduce_add                                ~15 + 1 add
      continuation_count  continuation.select(1,0).reduce_add    ~15
    """
    return (3) + (10) + (6) + (15) + (19) + (18) + (15)


def codepoint_ops_per_codepoint() -> int:
    """`mcn_codepoint_stats`: scalar per codepoint, 24 counters.

    Control/invalid ranges, the 16-way script chain, and the punctuation,
    digit and byte-lookup counters. Counting the branches along the average
    path (a script match lands early in the chain); a CJK or Hangul codepoint
    walks further and costs more.
    """
    special_counters = (4 + 4) + 2 + (6 + 6) + 2 + (2 + 2) + 2  # 6 counters
    script_chain = 8 + 6  # ~7 range tests tried before a match, then the match
    tail_counters = 4 + 6 + 8 + 5 + 2 + 2
    transitions = 2
    return special_counters + script_chain + tail_counters + transitions + 1


def report() -> None:
    raw = raw_stats_ops_per_vector()
    utf8 = utf8_count_ops_per_vector()
    cp = codepoint_ops_per_codepoint()

    rows = [
        # name, ops per unit, bytes per unit, unit
        ("mcn_text_scan (raw_stats, ASCII/binary)", raw, LANES, "byte"),
        ("mcn_utf8_stats (utf8_count, per vector)", utf8, LANES, "byte"),
        # A codepoint arrives as 4 bytes of UTF-32LE from the Python layer.
        ("mcn_codepoint_stats (UTF-32LE input)", cp, 4, "codepoint"),
        ("mcn_byte_stats (serial histogram)", 3, 1, "byte"),
    ]

    print(f"{'kernel':44s} {'flops/byte':>10s}  gate")
    print(f"{'-' * 44} {'-' * 10}  {'-' * 22}")
    for name, ops, unit_bytes, unit in rows:
        intensity = ops / unit_bytes
        verdict = "candidate" if intensity > 2.0 else "CPU only: memory-bound"
        print(f"{name:44s} {intensity:10.2f}  {verdict}")
    print()
    print(f"SIMD lanes assumed: {LANES}")


if __name__ == "__main__":
    report()
