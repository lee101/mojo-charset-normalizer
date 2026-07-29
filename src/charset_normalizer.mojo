"""Byte and Unicode scoring kernels for charset detection."""

from std.algorithm import parallelize
from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BYTE_W = simd_width_of[DType.uint8]()
comptime RAW_WORKERS = 4
comptime RAW_PARALLEL_THRESHOLD = 32 * 1024 * 1024


@export("mcn_byte_stats")
def mcn_byte_stats(
    src_addr: Int, n: Int, histogram_addr: Int, stats_addr: Int
) abi("C") -> Int:
    if n < 0 or histogram_addr == 0 or stats_addr == 0:
        return -1
    if n > 0 and src_addr == 0:
        return -1
    var histogram = U64Ptr(unsafe_from_address=histogram_addr)
    var stats = I64Ptr(unsafe_from_address=stats_addr)
    for i in range(256):
        histogram[i] = 0
    for i in range(8):
        stats[i] = 0
    if n <= 0:
        return 0

    var src = BPtr(unsafe_from_address=src_addr)
    for i in range(n):
        var b = Int(src[i])
        histogram[b] += 1
        if b < 128:
            stats[0] += 1
        else:
            stats[1] += 1
        if b == 0:
            stats[2] += 1
        if (b < 32 and b != 9 and b != 10 and b != 12 and b != 13) or b == 127:
            stats[3] += 1
        if b == 9 or b == 10 or b == 12 or b == 13 or b == 32:
            stats[4] += 1
        if b >= 32 and b < 127:
            stats[5] += 1
        if b >= 128 and b < 160:
            stats[6] += 1
        if b == 10:
            stats[7] += 1
    return 0


@export("mcn_utf8_stats")
def mcn_utf8_stats(src_addr: Int, n: Int, stats_addr: Int) abi("C") -> Int:
    if n < 0 or stats_addr == 0 or (n > 0 and src_addr == 0):
        return -1
    var stats = I64Ptr(unsafe_from_address=stats_addr)
    for j in range(3):
        stats[j] = 0
    if n <= 0:
        return 0

    var src = BPtr(unsafe_from_address=src_addr)
    return utf8_stats(src, n, stats)


def ascii_prefix(src: BPtr, n: Int) -> Int:
    var i = 0
    while i + BYTE_W <= n:
        var chunk = src.load[width=BYTE_W](i)
        if (
            chunk.ge(SIMD[DType.uint8, BYTE_W](128))
            .select(
                SIMD[DType.uint8, BYTE_W](1),
                SIMD[DType.uint8, BYTE_W](0),
            )
            .reduce_add()
            != 0
        ):
            break
        i += BYTE_W
    while i < n and src[i] < 128:
        i += 1
    return i


def utf8_stats_scalar(src: BPtr, n: Int, stats: I64Ptr) -> Int:
    var i = 0
    while i < n:
        var ascii_count = ascii_prefix(src + i, n - i)
        if ascii_count:
            stats[0] += Int64(ascii_count)
            i += ascii_count
            if i >= n:
                break
        var b0 = Int(src[i])
        var needed: Int
        var cp: Int
        var minimum: Int
        if b0 >= 0xC2 and b0 <= 0xDF:
            needed = 1
            cp = b0 & 0x1F
            minimum = 0x80
        elif b0 >= 0xE0 and b0 <= 0xEF:
            needed = 2
            cp = b0 & 0x0F
            minimum = 0x800
        elif b0 >= 0xF0 and b0 <= 0xF4:
            needed = 3
            cp = b0 & 0x07
            minimum = 0x10000
        else:
            return i

        if i + needed >= n:
            return i
        for j in range(1, needed + 1):
            var continuation = Int(src[i + j])
            if continuation < 0x80 or continuation > 0xBF:
                return i + j
            cp = (cp << 6) | (continuation & 0x3F)

        if cp < minimum or cp > 0x10FFFF or (cp >= 0xD800 and cp <= 0xDFFF):
            return i
        stats[0] += 1
        stats[1] += 1
        stats[2] += Int64(needed + 1)
        i += needed + 1
    return n


def valid_utf8_byte(src: BPtr, i: Int) -> Bool:
    var current = Int(src[i])
    var continuation = current >= 0x80 and current <= 0xBF
    var required = False
    if i >= 1:
        var previous = Int(src[i - 1])
        required = previous >= 0xC2 and previous <= 0xF4
        if previous == 0xE0 and current < 0xA0:
            return False
        if previous == 0xED and current >= 0xA0:
            return False
        if previous == 0xF0 and current < 0x90:
            return False
        if previous == 0xF4 and current >= 0x90:
            return False
    if i >= 2:
        var previous2 = Int(src[i - 2])
        required = required or (previous2 >= 0xE0 and previous2 <= 0xF4)
    if i >= 3:
        var previous3 = Int(src[i - 3])
        required = required or (previous3 >= 0xF0 and previous3 <= 0xF4)
    var valid_kind = (
        current < 0x80
        or continuation
        or (current >= 0xC2 and current <= 0xF4)
    )
    return valid_kind and continuation == required


def utf8_stats(src: BPtr, n: Int, stats: I64Ptr) -> Int:
    var ascii_count = ascii_prefix(src, n)
    if ascii_count == n:
        stats[0] = Int64(n)
        return n

    var high_count: Int64 = 0
    var continuation_count: Int64 = 0
    var i = 0
    while i < min(3, n):
        if not valid_utf8_byte(src, i):
            return utf8_stats_scalar(src, n, stats)
        var value = Int(src[i])
        if value >= 0x80:
            high_count += 1
        if value >= 0x80 and value <= 0xBF:
            continuation_count += 1
        i += 1

    var zero = SIMD[DType.uint8, BYTE_W](0)
    var one = SIMD[DType.uint8, BYTE_W](1)
    var false_mask = SIMD[DType.bool, BYTE_W](fill=False)
    var c80 = SIMD[DType.uint8, BYTE_W](0x80)
    var c90 = SIMD[DType.uint8, BYTE_W](0x90)
    var ca0 = SIMD[DType.uint8, BYTE_W](0xA0)
    var cc2 = SIMD[DType.uint8, BYTE_W](0xC2)
    var ce0 = SIMD[DType.uint8, BYTE_W](0xE0)
    var ced = SIMD[DType.uint8, BYTE_W](0xED)
    var cf0 = SIMD[DType.uint8, BYTE_W](0xF0)
    var cf4 = SIMD[DType.uint8, BYTE_W](0xF4)
    var cf5 = SIMD[DType.uint8, BYTE_W](0xF5)
    var invalid = False
    while i + BYTE_W <= n:
        var current = src.load[width=BYTE_W](i)
        var previous = src.load[width=BYTE_W](i - 1)
        var previous2 = src.load[width=BYTE_W](i - 2)
        var previous3 = src.load[width=BYTE_W](i - 3)
        var continuation = current.ge(c80) & current.lt(cc2)
        var required = (
            (previous.ge(cc2) & previous.lt(cf5))
            | (previous2.ge(ce0) & previous2.lt(cf5))
            | (previous3.ge(cf0) & previous3.lt(cf5))
        )
        var valid_kind = current.lt(c80) | continuation | (
            current.ge(cc2) & current.lt(cf5)
        )
        var bad_special = (
            (previous.eq(ce0) & current.lt(ca0))
            | (previous.eq(ced) & current.ge(ca0))
            | (previous.eq(cf0) & current.lt(c90))
            | (previous.eq(cf4) & current.ge(c90))
        )
        var bad = continuation.ne(required) | valid_kind.eq(false_mask) | bad_special
        if bad.select(one, zero).reduce_add() != 0:
            invalid = True
            break
        high_count += Int64(current.ge(c80).select(one, zero).reduce_add())
        continuation_count += Int64(continuation.select(one, zero).reduce_add())
        i += BYTE_W

    if invalid:
        return utf8_stats_scalar(src, n, stats)
    while i < n:
        if not valid_utf8_byte(src, i):
            return utf8_stats_scalar(src, n, stats)
        var value = Int(src[i])
        if value >= 0x80:
            high_count += 1
        if value >= 0x80 and value <= 0xBF:
            continuation_count += 1
        i += 1

    if (
        (n >= 1 and src[n - 1] >= 0xC2)
        or (n >= 2 and src[n - 2] >= 0xE0 and src[n - 2] <= 0xF4)
        or (n >= 3 and src[n - 3] >= 0xF0 and src[n - 3] <= 0xF4)
    ):
        return utf8_stats_scalar(src, n, stats)
    stats[0] = Int64(n) - continuation_count
    stats[1] = high_count - continuation_count
    stats[2] = high_count
    return n


def raw_stats(src: BPtr, n: Int, raw: I64Ptr):
    for j in range(8):
        raw[j] = 0
    var i = 0
    var zero = SIMD[DType.uint8, BYTE_W](0)
    var one = SIMD[DType.uint8, BYTE_W](1)
    while i + BYTE_W <= n:
        var values = src.load[width=BYTE_W](i)
        var ascii = values.lt(SIMD[DType.uint8, BYTE_W](128))
        var high = values.ge(SIMD[DType.uint8, BYTE_W](128))
        var nul = values.eq(zero)
        var control = (
            values.lt(SIMD[DType.uint8, BYTE_W](32))
            & values.ne(SIMD[DType.uint8, BYTE_W](9))
            & values.ne(SIMD[DType.uint8, BYTE_W](10))
            & values.ne(SIMD[DType.uint8, BYTE_W](12))
            & values.ne(SIMD[DType.uint8, BYTE_W](13))
        ) | values.eq(SIMD[DType.uint8, BYTE_W](127))
        var whitespace = (
            values.eq(SIMD[DType.uint8, BYTE_W](9))
            | values.eq(SIMD[DType.uint8, BYTE_W](10))
            | values.eq(SIMD[DType.uint8, BYTE_W](12))
            | values.eq(SIMD[DType.uint8, BYTE_W](13))
            | values.eq(SIMD[DType.uint8, BYTE_W](32))
        )
        var printable = values.ge(SIMD[DType.uint8, BYTE_W](32)) & values.lt(
            SIMD[DType.uint8, BYTE_W](127)
        )
        var c1 = values.ge(SIMD[DType.uint8, BYTE_W](128)) & values.lt(
            SIMD[DType.uint8, BYTE_W](160)
        )
        var newline = values.eq(SIMD[DType.uint8, BYTE_W](10))
        raw[0] += Int64(ascii.select(one, zero).reduce_add())
        raw[1] += Int64(high.select(one, zero).reduce_add())
        raw[2] += Int64(nul.select(one, zero).reduce_add())
        raw[3] += Int64(control.select(one, zero).reduce_add())
        raw[4] += Int64(whitespace.select(one, zero).reduce_add())
        raw[5] += Int64(printable.select(one, zero).reduce_add())
        raw[6] += Int64(c1.select(one, zero).reduce_add())
        raw[7] += Int64(newline.select(one, zero).reduce_add())
        i += BYTE_W
    while i < n:
        var b = Int(src[i])
        if b < 128:
            raw[0] += 1
        else:
            raw[1] += 1
        if b == 0:
            raw[2] += 1
        if (b < 32 and b != 9 and b != 10 and b != 12 and b != 13) or b == 127:
            raw[3] += 1
        if b == 9 or b == 10 or b == 12 or b == 13 or b == 32:
            raw[4] += 1
        if b >= 32 and b < 127:
            raw[5] += 1
        if b >= 128 and b < 160:
            raw[6] += 1
        if b == 10:
            raw[7] += 1
        i += 1


@export("mcn_text_scan")
def mcn_text_scan(
    src_addr: Int, n: Int, raw_addr: Int, utf8_addr: Int, scratch_addr: Int
) abi("C") -> Int:
    if (
        n < 0
        or raw_addr == 0
        or utf8_addr == 0
        or (n > 0 and src_addr == 0)
        or (n >= RAW_PARALLEL_THRESHOLD and scratch_addr == 0)
    ):
        return -1
    var raw = I64Ptr(unsafe_from_address=raw_addr)
    var utf8 = I64Ptr(unsafe_from_address=utf8_addr)
    for j in range(8):
        raw[j] = 0
    for j in range(3):
        utf8[j] = 0
    if n <= 0:
        return 0

    var src = BPtr(unsafe_from_address=src_addr)
    if n >= RAW_PARALLEL_THRESHOLD:
        var scratch = I64Ptr(unsafe_from_address=scratch_addr)

        @parameter
        def scan_chunk(worker: Int):
            var start = n * worker // RAW_WORKERS
            var end = n * (worker + 1) // RAW_WORKERS
            raw_stats(src + start, end - start, scratch + worker * 8)

        parallelize[scan_chunk](RAW_WORKERS, RAW_WORKERS)
        for worker in range(RAW_WORKERS):
            for j in range(8):
                raw[j] += scratch[worker * 8 + j]
    else:
        raw_stats(src, n, raw)
    if raw[1] == 0:
        utf8[0] = Int64(n)
        return n
    return utf8_stats(src, n, utf8)


def is_latin_letter(cp: Int) -> Bool:
    return (
        (cp >= 0x41 and cp <= 0x5A)
        or (cp >= 0x61 and cp <= 0x7A)
        or (cp >= 0xC0 and cp <= 0x2AF)
        or (cp >= 0x1E00 and cp <= 0x1EFF)
    )


@export("mcn_codepoint_stats")
def mcn_codepoint_stats(src_addr: Int, n: Int, stats_addr: Int) abi("C") -> Int:
    if n < 0 or stats_addr == 0 or (n > 0 and src_addr == 0):
        return -1
    var stats = I64Ptr(unsafe_from_address=stats_addr)
    for j in range(24):
        stats[j] = 0
    if n <= 0:
        return 0

    var src = U32Ptr(unsafe_from_address=src_addr)
    var previous_script = 0
    for i in range(n):
        var cp = Int(src[i])
        stats[0] += 1
        var script = 0

        if (cp < 32 and cp != 9 and cp != 10 and cp != 12 and cp != 13) or (
            cp >= 0x7F and cp <= 0x9F
        ):
            stats[1] += 1
        if cp == 0xFFFD:
            stats[2] += 1
        if (
            (cp >= 0xE000 and cp <= 0xF8FF)
            or (cp >= 0xF0000 and cp <= 0xFFFFD)
            or (cp >= 0x100000 and cp <= 0x10FFFD)
        ):
            stats[3] += 1
        if cp >= 0xD800 and cp <= 0xDFFF:
            stats[4] += 1
        if (cp >= 0xFDD0 and cp <= 0xFDEF) or (cp & 0xFFFF) >= 0xFFFE:
            stats[5] += 1

        if is_latin_letter(cp):
            stats[6] += 1
            script = 1
        elif cp >= 0x370 and cp <= 0x3FF:
            stats[7] += 1
            script = 2
        elif cp >= 0x400 and cp <= 0x52F:
            stats[8] += 1
            script = 3
        elif cp >= 0x590 and cp <= 0x5FF:
            stats[9] += 1
            script = 4
        elif (cp >= 0x600 and cp <= 0x6FF) or (cp >= 0x750 and cp <= 0x77F):
            stats[10] += 1
            script = 5
        elif cp >= 0x900 and cp <= 0x97F:
            stats[11] += 1
            script = 6
        elif cp >= 0xE00 and cp <= 0xE7F:
            stats[12] += 1
            script = 7
        elif cp >= 0x3040 and cp <= 0x309F:
            stats[13] += 1
            script = 8
        elif cp >= 0x30A0 and cp <= 0x30FF:
            stats[14] += 1
            script = 9
        elif (
            (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x20000 and cp <= 0x2FA1F)
        ):
            stats[15] += 1
            script = 10
        elif (cp >= 0x1100 and cp <= 0x11FF) or (cp >= 0xAC00 and cp <= 0xD7AF):
            stats[16] += 1
            script = 11

        if cp >= 0x30 and cp <= 0x39:
            stats[17] += 1
        if cp == 9 or cp == 10 or cp == 12 or cp == 13 or cp == 32 or cp == 0xA0:
            stats[18] += 1
        if (
            (cp >= 0x21 and cp <= 0x2F)
            or (cp >= 0x3A and cp <= 0x40)
            or (cp >= 0x5B and cp <= 0x60)
            or (cp >= 0x7B and cp <= 0x7E)
            or (cp >= 0x2000 and cp <= 0x206F)
            or (cp >= 0x3000 and cp <= 0x303F)
        ):
            stats[19] += 1
        if cp == 0xC2 or cp == 0xC3 or cp == 0xD0 or cp == 0xD1 or cp == 0xE2:
            stats[20] += 1
        if cp >= 0xA0 and cp <= 0xBF and not is_latin_letter(cp):
            stats[21] += 1

        if script != 0:
            stats[22] += 1
            if previous_script != 0 and previous_script != script:
                stats[23] += 1
            previous_script = script
    return 0
