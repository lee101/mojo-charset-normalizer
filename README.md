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
streaming scans, including scalar remainder handling. Raw classification uses
multiple CPU workers only for substantially larger inputs; smaller inputs
remain serial because launch overhead loses at benchmark sizes. Large valid
UTF-8 payloads are scored from bounded samples and decoded lazily, while
single-byte legacy candidates are sampled before allocating full decoded
strings. Multibyte and BOM-less Unicode candidates are screened with bounded,
boundary-aware windows before a full strict decode. Candidate fingerprints are
cached instead of repeatedly hashing multi-megabyte strings.

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
