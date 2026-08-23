from __future__ import annotations

import codecs
from collections import Counter
import logging
import re
from os import PathLike
from typing import BinaryIO

from ._lib import codepoint_statistics, text_statistics
from .models import CharsetMatch, CharsetMatches, iana_name

logger = logging.getLogger("mojo_charset_normalizer")

_BOMS = (
    (b"\xff\xfe\x00\x00", "utf_32"),
    (b"\x00\x00\xfe\xff", "utf_32"),
    (b"\xef\xbb\xbf", "utf_8"),
    (b"\xff\xfe", "utf_16"),
    (b"\xfe\xff", "utf_16"),
)

_CANDIDATES = (
    "cp1252", "cp1250", "cp1257", "iso8859_15", "iso8859_1", "mac_roman",
    "cp1251", "koi8_r", "iso8859_5", "mac_cyrillic",
    "cp1253", "iso8859_7", "cp1255", "iso8859_8",
    "cp1256", "iso8859_6", "cp874", "tis_620",
    "cp932", "shift_jis", "euc_jp", "euc_jis_2004",
    "gb18030", "gbk", "big5", "euc_kr",
)
_SINGLE_BYTE_ENCODINGS = frozenset(_CANDIDATES[:18])
_UNDEFINED_BYTES = {
    encoding: tuple(
        value
        for value in range(256)
        if not bytes((value,)).decode(encoding, "ignore")
    )
    for encoding in _SINGLE_BYTE_ENCODINGS
}

_TARGET_SCRIPT = {
    "cp1251": 8, "koi8_r": 8, "iso8859_5": 8, "mac_cyrillic": 8,
    "cp1253": 7, "iso8859_7": 7,
    "cp1255": 9, "iso8859_8": 9,
    "cp1256": 10, "iso8859_6": 10,
    "cp874": 12, "tis_620": 12,
    "cp932": 15, "shift_jis": 15, "euc_jp": 15, "euc_jis_2004": 15,
    "gb18030": 15, "gbk": 15, "big5": 15,
    "euc_kr": 16,
}

_LATIN_ENCODINGS = {
    "cp1252", "cp1250", "cp1257", "iso8859_15", "iso8859_1", "mac_roman"
}
_CYRILLIC_FREQUENCY = "оеаинтсрвлкмдпуяызьбгчйхжюшцщэфъё"
_GREEK_FREQUENCY = "αεοιστνρηκπμλγυδωχφβξζψθ"
_ARABIC_FREQUENCY = "اليومنرتبدمكسةفوعهقحجشصطزخضظغثذءئىؤئ"
_COMMON_CJK = frozenset(
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会"
    "可主发年动同工也能下过子说产种面而方后多定行学法所民得经十之进着等部"
    "度家电力里如水高自理起小物现实量都体制机当使点从业本去性好开合因然前"
    "外天日社相全表间样关新内数正心反明看原利比质气第向道命变结解问意建月"
    "公无系情者最立代想已通提直题程展果料员常文总次品式活设管特长求老头路"
    "少图山统接知将组见计手期根论运指区强放先回任取南给色光门保治北百海口"
    "东金清美教花安身车真万每目走声完类名科信话空今集传土群石界林观影持音"
    "書読聞食校語女男母父雨川円気話何上下左右中年月火水木"
)


def _declared_encoding(payload: bytes | bytearray) -> str | None:
    zone = bytes(payload[:8192]).decode("ascii", "ignore")
    match = re.search(
        r"(?i)(?:coding\s*[:=]\s*|charset\s*=\s*[\"']?)([-\w.]+)", zone
    )
    if not match:
        return None
    try:
        return iana_name(match.group(1))
    except LookupError:
        return None


def _decode(payload: bytes | bytearray, encoding: str, bom: bool = False) -> str:
    codec = encoding
    if bom and encoding == "utf_8":
        codec = "utf_8_sig"
    return payload.decode(codec, "strict")


def _sample_text(text: str, steps: int, chunk_size: int) -> str:
    if steps <= 1 or len(text) <= steps * chunk_size:
        return text
    width = max(1, chunk_size)
    last = max(0, len(text) - width)
    return "".join(
        text[(last * index) // (steps - 1):(last * index) // (steps - 1) + width]
        for index in range(steps)
    )


def _sample_bytes(
    payload: bytes | bytearray, steps: int, chunk_size: int
) -> bytes | bytearray:
    if steps <= 1 or len(payload) <= steps * chunk_size:
        return payload
    width = max(1, chunk_size)
    last = max(0, len(payload) - width)
    return b"".join(
        payload[
            (last * index) // (steps - 1):
            (last * index) // (steps - 1) + width
        ]
        for index in range(steps)
    )


def _sample_code_units(
    payload: bytes | bytearray, steps: int, chunk_size: int, unit_size: int
) -> bytes | bytearray:
    width = max(1, chunk_size) * unit_size
    if steps <= 1 or len(payload) <= max(1, steps) * width:
        return payload
    last = max(0, len(payload) - width)
    last -= last % unit_size
    chunks: list[bytes | bytearray] = []
    for index in range(max(1, steps)):
        start = (last * index) // (steps - 1)
        start -= start % unit_size
        chunks.append(payload[start:start + width])
    return b"".join(chunks)


def _sample_multibyte(
    payload: bytes | bytearray, encoding: str, steps: int, chunk_size: int
) -> str:
    width = max(1, chunk_size) * 4
    if steps <= 1 or len(payload) <= max(1, steps) * width:
        return payload.decode(encoding, "strict")
    last = max(0, len(payload) - width)
    chunks: list[str] = []
    for index in range(max(1, steps)):
        start = (last * index) // (steps - 1)
        chunks.append(
            payload[start:start + width].decode(encoding, "ignore")[:chunk_size]
        )
    return "".join(chunks)


def _sample_utf8(payload: bytes | bytearray, steps: int, chunk_size: int) -> str:
    width = max(1, chunk_size)
    window = width * 4
    if steps <= 1 or len(payload) <= max(1, steps) * window:
        return payload.decode("utf_8", "strict")
    last = max(0, len(payload) - window)
    chunks: list[str] = []
    for index in range(max(1, steps)):
        start = (last * index) // (steps - 1)
        while start < len(payload) and 0x80 <= payload[start] <= 0xBF:
            start += 1
        end = min(len(payload), start + window)
        while end < len(payload) and 0x80 <= payload[end] <= 0xBF:
            end += 1
        chunks.append(payload[start:end].decode("utf_8", "strict")[:width])
    return "".join(chunks)


def _pattern_encoding(payload: bytes | bytearray) -> str | None:
    n = len(payload)
    if payload.find(b"\x00") < 0:
        return None
    if n >= 8 and n % 4 == 0:
        quarters = [bytes(payload[i::4]).count(b"\x00") / (n // 4) for i in range(4)]
        if quarters[1] > 0.8 and quarters[2] > 0.8 and quarters[3] > 0.8:
            return "utf_32_le"
        if quarters[0] > 0.8 and quarters[1] > 0.8 and quarters[2] > 0.8:
            return "utf_32_be"
    if n >= 4 and n % 2 == 0:
        even = bytes(payload[0::2]).count(b"\x00") / (n // 2)
        odd = bytes(payload[1::2]).count(b"\x00") / (n // 2)
        if odd > 0.65 and even < 0.2:
            return "utf_16_le"
        if even > 0.65 and odd < 0.2:
            return "utf_16_be"
    return None


def _language(text: str, stats, encoding: str) -> tuple[str, float]:
    counts = {
        "Latin": int(stats[6]), "Greek": int(stats[7]), "Cyrillic": int(stats[8]),
        "Hebrew": int(stats[9]), "Arabic": int(stats[10]), "Hindi": int(stats[11]),
        "Thai": int(stats[12]), "Japanese": int(stats[13] + stats[14]),
        "CJK": int(stats[15]), "Korean": int(stats[16]),
    }
    letters = sum(counts.values())
    if not letters:
        return "Unknown", 0.0
    dominant, count = max(counts.items(), key=lambda item: item[1])
    coherence = count / letters
    if dominant == "Latin":
        lowered = f" {text.lower()} "
        profiles = (
            ("French", (" le ", " la ", " les ", " des ", " une ", " est ", "é", "à")),
            ("German", (" der ", " die ", " und ", " ist ", " das ", "ä", "ß")),
            ("Spanish", (" el ", " la ", " de ", " que ", " los ", "ñ", "¿")),
            ("Portuguese", (" de ", " que ", " não ", " uma ", "ção", "ã")),
            ("Italian", (" il ", " che ", " di ", " una ", " gli ")),
            ("Polish", (" nie ", " jest ", " się ", " oraz ", "ł", "ż", "ź")),
        )
        scores = [(name, sum(lowered.count(token) for token in tokens)) for name, tokens in profiles]
        name, score = max(scores, key=lambda item: item[1])
        return (name if score else "English"), min(1.0, coherence * (0.5 + score / 12))
    if dominant == "CJK":
        if counts["Japanese"]:
            return "Japanese", coherence
        if encoding in {"cp932", "shift_jis", "euc_jp", "euc_jis_2004"}:
            return "Japanese", coherence
        return "Chinese", coherence
    return dominant, coherence


def _score(text: str, encoding: str) -> tuple[float, list[tuple[str, float]]]:
    stats = codepoint_statistics(text)
    total = max(1, int(stats[0]))
    severe = int(stats[1] * 2 + stats[2] * 5 + stats[3] * 3 + stats[4] * 5 + stats[5] * 5)
    chaos = severe / total
    script_letters = int(stats[22])
    switches = int(stats[23])
    japanese_letters = int(stats[13] + stats[14] + stats[15])
    if script_letters > 8 and japanese_letters / script_letters < 0.5:
        chaos += max(0.0, switches / script_letters - 0.12) * 0.25
    target = _TARGET_SCRIPT.get(encoding)
    if target is not None and script_letters:
        target_count = int(stats[target])
        if encoding in {"cp932", "shift_jis", "euc_jp", "euc_jis_2004"}:
            target_count += int(stats[13] + stats[14])
        target_share = target_count / script_letters
        if target_share < 0.55:
            chaos += (0.55 - target_share) * 0.7
    elif encoding in _LATIN_ENCODINGS and script_letters and int(stats[6]) / script_letters < 0.55:
        chaos += 0.3
    chaos += min(0.2, int(stats[21]) / total * 0.8)
    if encoding in _CANDIDATES:
        chaos += 0.01
        counts = Counter(text.lower())
        if encoding in _LATIN_ENCODINGS:
            non_ascii = total - len(text.encode("ascii", "ignore"))
            if non_ascii / total > 0.45:
                chaos += 0.25
        elif target == 8:
            letters = sum(counts[char] for char in _CYRILLIC_FREQUENCY)
            if letters:
                rank = {char: index for index, char in enumerate(_CYRILLIC_FREQUENCY)}
                weighted = sum(
                    counts[char] * rank[char] for char in _CYRILLIC_FREQUENCY
                )
                chaos += 0.08 * weighted / letters / len(_CYRILLIC_FREQUENCY)
        elif target == 7:
            letters = sum(counts[char] for char in _GREEK_FREQUENCY)
            if letters:
                rank = {char: index for index, char in enumerate(_GREEK_FREQUENCY)}
                weighted = sum(counts[char] * rank[char] for char in _GREEK_FREQUENCY)
                chaos += 0.08 * weighted / letters / len(_GREEK_FREQUENCY)
        elif target == 10:
            letters = sum(counts[char] for char in _ARABIC_FREQUENCY)
            if letters:
                rank = {char: index for index, char in enumerate(_ARABIC_FREQUENCY)}
                weighted = sum(counts[char] * rank[char] for char in _ARABIC_FREQUENCY)
                chaos += 0.08 * weighted / letters / len(_ARABIC_FREQUENCY)
        if target == 15:
            cjk = int(stats[15])
            uncommon = sum(
                count for char, count in counts.items()
                if 0x3400 <= ord(char) <= 0x9FFF and char not in _COMMON_CJK
            )
            if cjk:
                chaos += 0.14 * uncommon / cjk
            kana = int(stats[13] + stats[14])
            if encoding in {"gb18030", "gbk", "big5"} and kana:
                chaos += min(0.12, 0.3 * kana / total)
            if encoding in {"cp932", "shift_jis", "euc_jp", "euc_jis_2004"} and cjk and not kana:
                chaos += 0.04
    language, coherence = _language(text, stats, encoding)
    languages = [] if language == "Unknown" else [(language, round(coherence, 4))]
    return round(min(1.0, max(0.0, chaos)), 4), languages


def _single_match(
    payload: bytes | bytearray,
    encoding: str,
    bom: bool,
    declared: str | None,
    steps: int = 5,
    chunk_size: int = 512,
) -> CharsetMatches:
    if (
        encoding == "ascii"
        and len(payload) > max(1, steps) * max(1, chunk_size)
    ):
        last = max(0, len(payload) - chunk_size)
        starts = [
            (last * index) // (steps - 1) if steps > 1 else 0
            for index in range(max(1, steps))
        ]
        sample = b"".join(bytes(payload[start:start + chunk_size]) for start in starts)
        text = sample.decode("ascii", "strict")
        chaos, languages = _score(text, "ascii")
        return CharsetMatches(
            [CharsetMatch(payload, encoding, chaos, bom, languages, None, declared)]
        )
    if (
        encoding == "utf_8"
        and not bom
        and len(payload) > max(1, steps) * max(1, chunk_size) * 4
    ):
        sample = _sample_utf8(payload, steps, chunk_size)
        chaos, languages = _score(sample, "utf_8")
        return CharsetMatches(
            [CharsetMatch(payload, encoding, chaos, bom, languages, None, declared)]
        )
    try:
        text = _decode(payload, encoding, bom)
    except (LookupError, UnicodeDecodeError):
        return CharsetMatches()
    chaos, languages = _score(
        _sample_text(text, steps, chunk_size), iana_name(encoding)
    )
    return CharsetMatches(
        [CharsetMatch(payload, encoding, chaos, bom, languages, text, declared)]
    )


def _bomless_unicode_match(
    payload: bytes | bytearray,
    threshold: float,
    declared: str | None,
    steps: int,
    chunk_size: int,
    nul_count: int,
) -> CharsetMatches:
    candidates: list[str] = []
    pattern = _pattern_encoding(payload) if nul_count else None
    if pattern:
        candidates.append(pattern)
    if len(payload) >= 8 and len(payload) % 4 == 0:
        candidates.extend(("utf_32_le", "utf_32_be"))
    if len(payload) >= 4 and len(payload) % 2 == 0:
        candidates.extend(("utf_16_le", "utf_16_be"))

    nul_ratio = nul_count / len(payload)
    ranked: list[tuple[float, float, str, list[tuple[str, float]]]] = []
    for encoding in dict.fromkeys(candidates):
        if encoding.startswith("utf_32") and nul_ratio < 0.35:
            continue
        unit_size = 4 if encoding.startswith("utf_32") else 2
        try:
            sample = _sample_code_units(payload, steps, chunk_size, unit_size).decode(
                encoding, "strict"
            )
        except UnicodeDecodeError:
            continue
        stats = codepoint_statistics(sample)
        total = max(1, int(stats[0]))
        recognized = (
            int(stats[22]) + int(stats[17]) + int(stats[18]) + int(stats[19])
        ) / total
        if recognized < 0.72:
            continue
        chaos, languages = _score(sample, encoding)
        cjk = int(stats[15])
        common = sum(sample.count(char) for char in _COMMON_CJK) if cjk else 0
        if (
            encoding.startswith("utf_16")
            and nul_ratio <= 0.02
            and not int(stats[13] + stats[14])
            and (not cjk or common / cjk <= 0.25)
        ):
            continue
        if cjk and not int(stats[13] + stats[14]):
            chaos += 0.2 * max(0, cjk - common) / cjk
        ranked.append((chaos, -recognized, encoding, languages))
    if not ranked:
        return CharsetMatches()
    chaos, _, encoding, languages = min(ranked)
    if chaos >= threshold:
        return CharsetMatches()
    try:
        text = payload.decode(encoding, "strict")
    except UnicodeDecodeError:
        return CharsetMatches()
    return CharsetMatches(
        [CharsetMatch(payload, encoding, chaos, False, languages, text, declared)]
    )


def from_bytes(
    sequences: bytes | bytearray,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.2,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    if not isinstance(sequences, (bytes, bytearray)):
        raise TypeError(f"Expected object of type bytes or bytearray, got: {type(sequences)}")
    if not sequences:
        return CharsetMatches([CharsetMatch(sequences, "utf_8", 0.0, False, [], "")])

    declared = _declared_encoding(sequences) if preemptive_behaviour else None
    isolation = {iana_name(value) for value in cp_isolation or []}
    exclusion = {iana_name(value) for value in cp_exclusion or []}

    def allowed(encoding: str) -> bool:
        canonical = iana_name(encoding)
        return (not isolation or canonical in isolation) and canonical not in exclusion

    for mark, encoding in _BOMS:
        if sequences.startswith(mark) and allowed(encoding):
            return _single_match(
                sequences, encoding, True, declared, steps, chunk_size
            )

    raw_stats, utf8 = text_statistics(sequences)
    valid_utf8, _, _, non_ascii = utf8
    if (
        valid_utf8
        and raw_stats[2] == 0
        and raw_stats[3] / len(sequences) < threshold
    ):
        encoding = "ascii" if non_ascii == 0 else "utf_8"
        if allowed(encoding):
            match = _single_match(
                sequences, encoding, False, declared, steps, chunk_size
            )
            if match and (match.best().chaos < threshold or enable_fallback):
                return match

    if (
        raw_stats[2] / len(sequences) <= 0.01
        and raw_stats[3] / len(sequences) >= 0.08
        and raw_stats[6] / len(sequences) >= 0.08
    ):
        return CharsetMatches()

    unicode_match = _bomless_unicode_match(
        sequences, threshold, declared, steps, chunk_size, int(raw_stats[2])
    )
    if unicode_match and allowed(unicode_match.best().encoding):
        return unicode_match

    if raw_stats[2] and raw_stats[2] / len(sequences) > 0.01:
        return CharsetMatches()
    if raw_stats[3] / len(sequences) >= threshold:
        return CharsetMatches()
    if (
        raw_stats[3] / len(sequences) >= 0.08
        and raw_stats[6] / len(sequences) >= 0.08
    ):
        return CharsetMatches()

    candidates = list(_CANDIDATES)
    if declared:
        candidates.insert(0, declared)
    if isolation:
        candidates = list(isolation)

    results = CharsetMatches()
    seen: set[str] = set()
    byte_sample = _sample_bytes(sequences, steps, chunk_size)
    for encoding in candidates:
        canonical = iana_name(encoding)
        if canonical in seen or not allowed(canonical):
            continue
        seen.add(canonical)
        if canonical in _SINGLE_BYTE_ENCODINGS:
            try:
                sample = byte_sample.decode(canonical, "strict")
            except (LookupError, UnicodeDecodeError):
                continue
            chaos, languages = _score(sample, canonical)
            text = None
            decoded_length_hint = len(sequences)
        else:
            try:
                sample = _sample_multibyte(
                    sequences, canonical, steps, chunk_size
                )
            except (LookupError, UnicodeDecodeError):
                continue
            chaos, languages = _score(sample, canonical)
            text = None
            decoded_length_hint = None
        if (
            canonical not in {"cp932", "shift_jis", "euc_jp", "euc_jis_2004",
                              "gb18030", "gbk", "big5", "euc_kr"}
            and raw_stats[1] / len(sequences) > 0.6
        ):
            chaos = round(min(1.0, chaos + 0.06), 4)
        if canonical == declared:
            chaos = max(0.0, chaos - 0.03)
        if chaos >= threshold:
            continue
        if text is None:
            if canonical in _SINGLE_BYTE_ENCODINGS:
                if any(
                    sequences.find(bytes((value,))) >= 0
                    for value in _UNDEFINED_BYTES[canonical]
                ):
                    continue
            else:
                try:
                    validated = sequences.decode(canonical, "strict")
                except UnicodeDecodeError:
                    continue
                decoded_length_hint = len(validated)
                text = validated
        results.append(
            CharsetMatch(
                sequences,
                canonical,
                chaos,
                False,
                languages,
                text,
                declared,
                sample,
                decoded_length_hint,
            )
        )
        if explain:
            logger.info("%s passed with chaos %.3f", canonical, chaos)

    if not results and enable_fallback and declared and allowed(declared):
        try:
            text = sequences.decode(declared, "strict")
            results.append(
                CharsetMatch(sequences, declared, threshold, False, [], text, declared)
            )
        except (LookupError, UnicodeDecodeError):
            pass
    return results


def from_fp(
    fp: BinaryIO,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.2,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    return from_bytes(
        fp.read(), steps, chunk_size, threshold, cp_isolation, cp_exclusion,
        preemptive_behaviour, explain, language_threshold, enable_fallback,
    )


def from_path(
    path: str | bytes | PathLike,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.2,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = True,
) -> CharsetMatches:
    with open(path, "rb") as fp:
        return from_fp(
            fp, steps, chunk_size, threshold, cp_isolation, cp_exclusion,
            preemptive_behaviour, explain, language_threshold, enable_fallback,
        )


def is_binary(
    fp_or_path_or_payload: PathLike | str | BinaryIO | bytes,
    steps: int = 5,
    chunk_size: int = 512,
    threshold: float = 0.2,
    cp_isolation: list[str] | None = None,
    cp_exclusion: list[str] | None = None,
    preemptive_behaviour: bool = True,
    explain: bool = False,
    language_threshold: float = 0.1,
    enable_fallback: bool = False,
) -> bool:
    kwargs = dict(
        steps=steps, chunk_size=chunk_size, threshold=threshold,
        cp_isolation=cp_isolation, cp_exclusion=cp_exclusion,
        preemptive_behaviour=preemptive_behaviour, explain=explain,
        language_threshold=language_threshold, enable_fallback=enable_fallback,
    )
    if isinstance(fp_or_path_or_payload, (str, PathLike)):
        return not from_path(fp_or_path_or_payload, **kwargs)
    if isinstance(fp_or_path_or_payload, (bytes, bytearray)):
        return not from_bytes(fp_or_path_or_payload, **kwargs)
    return not from_fp(fp_or_path_or_payload, **kwargs)
