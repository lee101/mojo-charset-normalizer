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

There is no GPU path. The large kernels are low-arithmetic-intensity streaming
byte scans, while Unicode scoring operates on small samples; device transfer
and launch overhead would dominate both.

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
