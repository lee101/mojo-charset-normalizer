# mojo-charset-normalizer

Charset detection and normalization with the byte-scanning core implemented in
[Mojo](https://www.modular.com/mojo). The Python API follows
`charset-normalizer` for the covered subset, so migration normally changes only
the import:

```python
from mojo_charset_normalizer import from_bytes

payload = "Καλημέρα κόσμε".encode("utf-8")
match = from_bytes(payload).best()

print(match.encoding)          # utf_8
print(str(match))              # Καλημέρα κόσμε
print(match.output("utf_8"))   # normalized UTF-8 bytes
```

This is a standalone implementation. `charset-normalizer` is installed only as
a development dependency for parity tests and benchmarks; the runtime does not
import or call it.

## Coverage

The tested upstream-compatible surface includes:

- `from_bytes`, `from_fp`, and `from_path`, with the same signatures;
- `is_binary` and the chardet-compatible `detect` result dictionary;
- `CharsetMatch` properties, alias lookup, and `output()` normalization;
- ASCII, UTF-8 (with and without a BOM), UTF-16, UTF-32, and the tested
  BOM-less UTF-16 variants;
- CP1251, KOI8-R, ISO-8859-7/CP1253, CP1256, Shift-JIS/CP932, and EUC-JP.

The package also exposes useful native primitives: `byte_frequencies`,
`entropy`, and `is_valid_utf8`.

This is not an exhaustive replacement for every codec, API detail, or
statistical model in `charset-normalizer`. The command-line interface,
upstream extension modules, plugin hooks, and upstream's full per-language
frequency corpus are not covered. Additional legacy candidates exist in the
implementation but are not claimed as supported until parity tests cover them.
Language labels and coherence scores are deliberately lightweight heuristics.
For byte-compatible encodings that decode to the same text, the chosen alias
may differ; normalization remains identical. The test suite checks the covered
Unicode and legacy families against `charset-normalizer` 3.4.9 on the same
payloads and checks the native UTF-8 validator against strict Python decoding,
including overlong, surrogate, out-of-range, and truncated sequences.

## Install

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

After `pixi install`, `pixi run build` writes the native library under `dist/`.
Run the example above with `pixi run python`. Imports rebuild the library when
the Mojo source is newer. A packaged deployment can point
`MOJO_CHARSET_NORMALIZER_LIB` at a prebuilt shared library.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30GHz,
Linux 6.8.0-147-generic, Python 3.13.14, and `charset-normalizer` 3.4.9. Times
are the best of five runs on identical payloads; the "before" column is the
previous revision measured on the same machine in the same session.

| case | Mojo port before | Mojo port after | upstream | speedup |
| --- | ---: | ---: | ---: | ---: |
| ASCII detection, 8 MB | 5.79 ms | 4.50 ms | 19.73 ms | 4.38x |
| UTF-8 detection, 8 MB | 8.42 ms | 7.11 ms | 19.54 ms | 2.75x |
| CP1251 detection, 2 MB | 10.40 ms | 9.94 ms | 35.91 ms | 3.61x |
| Shift-JIS detection, 2 MB | 44.66 ms | 43.45 ms | 73.15 ms | 1.68x |
| binary classification, 8 MB | 5.32 ms | 4.30 ms | 18.07 ms | 4.20x |

The serial-chain pass retires the byte-class and UTF-8 scans faster in three
ways, all verified by `pixi run bench` at real caller sizes:

- `raw_stats` keeps eight independent per-lane accumulators instead of doing
  eight horizontal `reduce_add` per vector and a memory read-modify-write on the
  output counters. The accumulators are 16-bit lanes flushed every 1024 vectors
  (each lane then holds at most 1024 counts, so the deferred reduction cannot
  wrap). This is the change behind the ASCII and binary rows.
- `ascii_prefix` and the UTF-8 validator's "any bad lane" test now use
  `reduce_or` directly rather than materialising a select-plus-`reduce_add`.
- Loop-invariant SIMD constants are hoisted out of both scans, and the
  byte-class scan uses one `cast` instead of a `select` per vector.

The multi-accumulator rewrite changes the summation order of the byte-class
counters. Those counters are integers, so the totals are still exact and the
benchmark and parity results are bit-identical; a floating-point reduction of
the same kind could differ in the last bits.

Tested and rejected (measured slower or neutral, so not kept): manual 2x
unrolling of the scans, 16-bit lane accumulators in the UTF-8 loop, and hoisting
the `mcn_codepoint_stats` counters into locals (the compiler already registers
them). `@parameter for` and `@unroll` are not available on the pinned compiler,
so compile-time unrolling is not used.

SIMD byte classification and structural UTF-8 validation accelerate the
streaming scans, including scalar remainder handling. Large valid UTF-8 payloads
are scored from bounded samples and decoded lazily, while single-byte legacy
candidates are sampled before allocating full decoded strings. Multibyte and
BOM-less Unicode candidates are screened with bounded, boundary-aware windows
before a full strict decode. Candidate fingerprints are cached instead of
repeatedly hashing multi-megabyte strings.

### Parallel raw scan: measured and rejected

A parallel variant of the raw byte-class scan was built and benchmarked, then
removed. It is not in the shipped library. `mcn_text_scan` makes one serial
`raw_stats` pass at every size, and its `scratch` argument is accepted for ABI
compatibility but not read.

The variant split the buffer into 16 equal chunks and ran `raw_stats` on each
chunk with `parallelize` from `max.algorithm` (worker counts 1, 2 and 4, never
more than 4), merged the 8 counters per chunk afterwards, and kept the serial
path below a size threshold.

**Where it wins.** The gain appears only in one narrow step. At 32 MB the
parallel path was about 3x faster than serial; below it, it was never faster.
Numbers are the best of 13 kernel runs with the process pinned to four CPUs
(`taskset -c 0-3`), milliseconds per call, ASCII payload:

| input size | serial | 1 worker | 2 workers | 4 workers |
| ---: | ---: | ---: | ---: | ---: |
| 8 MB | 5.3 | 4.4 | 4.4 | 4.4 |
| 16 MB | 8.9 | 8.8 | 8.7 | 9.0 |
| 24 MB | 13.9 | 13.8 | 13.8 | 13.7 |
| 30 MB | 17.3 | 17.2 | 17.2 | 17.2 |
| 32 MB | 18.9 | 18.5 | 11.2 | 6.0 |
| 40 MB | 23.1 | 23.4 | 14.8 | 10.5 |
| 64 MB | 38.6 | 37.7 | 23.3 | 13.5 |

End to end through `from_bytes`, the same payload went from 19.5 ms serial
around 32 MB to 6.5 ms with four workers, and from 39 ms to 12-13 ms at 64 MB.
The worker scaling kept improving up to 4, so the cap was not the limit; more
than 4 workers were never measured, because the rules here forbade it.

The 32 MB step is where the parallel path switched on, not a CPU effect: the
serial numbers are a straight line through that point (24 MB: 13.9 ms, 30 MB:
17.3 ms, 32 MB: 18.9 ms), and splitting the scan into 4 or 16 chunks with no
threads at all stayed neutral. On a shared host the cost of a parallel call
scales with the number of CPUs the process can see (about 60 us per worker to
launch), so without pinning a call costs about 4.5 ms of launch time and wins
nothing below roughly 64 MB.

**Why it was removed.** At 32 MB and above, the parallel path reproducibly
killed the interpreter when the process exited: 4 of 4 runs through
`text_statistics` at 32 MB and 3 of 3 `from_bytes` runs at 30-64 MB, while the
serial build crashed in 0 of 8 runs on the same payloads. The crash is at
interpreter shutdown, after every result had been computed and returned
correctly (identical counters), as a segmentation fault in CPython's own GC
(`_PyGCHead_NEXT(gc=0x0)` in `update_refs`, called from `Py_FinalizeEx`) or as
a glibc `corrupted size vs. prev_size` abort in the
`libKGENCompilerRTShared.so` atexit handler. A bare `ctypes` call to the same
parallel kernel from a standalone script did not reproduce it, so the trigger
needs the package's normal call path; that is exactly the path callers use, and
a library that crashes the host process at exit is worse than no parallelism.

The variant was reverted rather than shipped behind a higher threshold, because
the crash covered every size where the parallel path could win, so raising the
threshold would only have moved the same crash to larger inputs. The test suite
keeps `test_text_scan_large_inputs_match_numpy`, which checks the serial scan at
31 MB and just above 32 MB against numpy counts, so the boundary stays covered.
`bench/ab.py` measures a 40 MB `mcn_text_scan` case, so re-testing a future
variant takes one command. `parallelize` from `max.algorithm` does compile and
run on this toolchain; the problem is the shutdown crash, not the import.

### GPU path: measured and rejected

There is no GPU path in the shipped library. A full device path for the two
heaviest kernels was built, verified for exact output parity with the CPU
kernels, measured at the sizes real callers pass, and then removed because it
lost on every case at every size.

**The gate.** A GPU only pays when a kernel does enough arithmetic per byte to
hide the copy. `pixi run python bench/intensity.py` counts the vector
operations in each hot loop and divides by the bytes one iteration consumes
(`perf` counters are blocked on this host, `perf_event_paranoid=4`):

| kernel | flops/byte | gate |
| --- | ---: | --- |
| `mcn_text_scan` (`raw_stats`, ASCII/binary) | 1.16 | CPU only: memory-bound |
| `mcn_utf8_stats` (`utf8_count`, per vector) | 2.69 | candidate |
| `mcn_codepoint_stats` (UTF-32LE input) | 18.50 | candidate |
| `mcn_byte_stats` (serial histogram) | 3.00 | candidate |

`simd_width_of[DType.uint8]()` is 32 on this compiler, and that is what decides
the byte scan: 37 counted vector ops per 32-byte iteration is 1.16 flops per
byte, under the bar. **CPU only: memory-bound** is therefore the correct
disposition for `mcn_text_scan` itself, before any timing. The other three
clear it, at 2.69 to 18.50, so they were the ones worth a device — and the
measurement below is why none of them got one.

**What was built.** `DeviceContext` from `max.gpu.host` with `block_idx` and
`thread_idx` from `max.gpu` on `mojo ==1.2.0.dev2026092605`. Each kernel copies
its input to the device once, runs a scan launch over the whole grid, folds the
per-block partials with one small launch (no shared memory, no atomics), and
copies results back once. The grid covers the whole input and the short final
block is handled by a per-item bounds check, which was tested at every length
from 0 to 40 bytes and at block boundaries. Counters are integers, so parity
with the CPU is exact, not approximate: 0 mismatches over those tails, at
64 KB and 65,537 bytes, and over 63 codepoint texts including script-transition
cases. Device allocations stayed under 2 GB: the largest was the 64 MB payload
plus 537 MB of per-thread partials.

**The measurement.** Shared host, RTX 5090 with 5498 MiB free at the time
(above the 4000 MiB gate), 72 CPUs, Linux 6.8.0-147-generic. Best of 7 runs,
CPU best-of-many amortised. Milliseconds per call:

| case | CPU | GPU | GPU vs CPU |
| --- | ---: | ---: | --- |
| `mcn_text_scan` ASCII 100 KB | 0.093 | 3.592 | 38.5x slower |
| `mcn_text_scan` ASCII 1 MB | 0.865 | 36.239 | 41.9x slower |
| `mcn_text_scan` ASCII 8 MB | 7.041 | 729.118 | 103.6x slower |
| `mcn_text_scan` ASCII 32 MB | 28.308 | 3478.838 | 122.9x slower |
| `mcn_text_scan` ASCII 64 MB | 56.807 | 7491.875 | 131.9x slower |
| `mcn_text_scan` UTF-8 8 MB | 10.102 | 743.122 | 73.6x slower |
| `mcn_text_scan` binary 8 MB | 6.873 | 936.897 | 136.3x slower |
| `mcn_codepoint_stats` CP1251, 2560 chars | 0.039 | 2.394 | 61.1x slower |
| `mcn_codepoint_stats` CP1251, 25,600 chars | 0.353 | 1.043 | 3.0x slower |
| `mcn_codepoint_stats` CP1251, 256k chars | 3.470 | 9.518 | 2.7x slower |
| `mcn_codepoint_stats` CP1251, 2.56M chars | 35.006 | 252.237 | 7.2x slower |
| `mcn_codepoint_stats` Shift-JIS, 2560 chars | 0.045 | 0.211 | 4.7x slower |
| `mcn_codepoint_stats` Shift-JIS, 256k chars | 4.015 | 9.531 | 2.4x slower |
| `mcn_codepoint_stats` Latin, 2560 chars | 0.042 | 2.399 | 56.9x slower |
| `mcn_codepoint_stats` Latin, 256k chars | 3.576 | 9.495 | 2.7x slower |

The launch-plus-copy floor on this host measured about 125 us per call, so
2,560 codepoints — the real default, `steps=5` and `chunk_size=512` — cannot
pay for a device trip even in principle, and they lose by 5-60x. Larger
codepoint samples (25,600 to 2.56M characters) still lose, because the scoring
pass is a branch-heavy scalar walk per codepoint and the device runs it at a
lower clock than the CPU, while a 2.56M-codepoint call also has to allocate
491 MB of per-thread partials. `mcn_utf8_stats` was not given a device pass of
its own: its `utf8_count` loop is a sequential validator with early exit, and
`mcn_text_scan` already runs it, so it was measured as part of that call.

The byte scan loses by a wider margin and the gap grows with size, which rules
out the transfer as the cause: the device sustained only 9-29 GB/s against the
CPU kernel's ~1.2 TB/s effective rate for an 8 MB payload. The device kernel
walks one byte per thread per step, so the 8 MB payload needs 32,768 blocks
(8.4M partial slots) where the CPU version does 16 bytes per vector instruction
with eight register-resident accumulators. A device version that loads vectors and
reduces them per block would narrow that, but it still has to beat a CPU pass
that already runs at memory-copy speed, and it still has to copy the payload in
and out.

**Verdict.** Reverted. The device never won at a size any caller passes, and it
won at no larger size either, so shipping it behind a size threshold would
only have added an untested branch.

**Tests skipped.** There is no GPU path, so the device tests skip rather than
run: `pytest tests` reports `66 passed, 1 skipped`, and the single skip is
`test_device_parity_when_present`, which requires `MOJO_CHARSET_EXPECT_GPU=1`
and a build carrying the `*_gpu` exports. A companion test asserts those
exports are absent, so a future attempt that forgets to revert cannot pass
silently. The GPU comparison benchmark is skipped on this build too:
`bench/gpu_ab.py` prints `GPU path not built: this library has no device
exports. No GPU rows to measure; skipped.` with exit status 0, so it does not
fail on a CPU-only host. `pixi run python bench/intensity.py` runs anywhere and
needs no device.

## How it works

Python owns every allocation. A payload is exposed as a contiguous `uint8`
buffer, then its address and length cross one `ctypes` call as 64-bit integers.
The Mojo export reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the C-ABI boundary. No Mojo
allocation crosses the FFI.

The hot detection path performs SIMD byte-class and strict UTF-8 scans. Legacy
candidates are sampled at the caller's `steps` and `chunk_size`, decoded by
Python's standard codecs, converted to contiguous UTF-32LE codepoints, and
scored in Mojo by controls, invalid scalar values, scripts, punctuation, and
script transitions. Full decoded strings are retained only for surviving
candidates or produced lazily. Python handles codec orchestration, language
hints, result objects, and normalization.

The exact 256-bin histogram is a separate kernel so ordinary detection does not
pay for it. All Mojo exports live in one compilation unit, and the shared
library is built once.

## License

MIT
