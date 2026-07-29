from __future__ import annotations

import codecs
import re
import unicodedata
from encodings.aliases import aliases
from typing import Iterator


def iana_name(name: str) -> str:
    try:
        return codecs.lookup(name).name.replace("-", "_")
    except LookupError:
        return name.lower().replace("-", "_")


class CharsetMatch:
    def __init__(
        self,
        payload: bytes | bytearray,
        guessed_encoding: str,
        mean_mess_ratio: float,
        has_sig_or_bom: bool,
        languages: list[tuple[str, float]],
        decoded_payload: str | None = None,
        preemptive_declaration: str | None = None,
    ):
        self._payload = payload
        self._encoding = iana_name(guessed_encoding)
        self._mean_mess_ratio = float(mean_mess_ratio)
        self._languages = languages
        self._has_sig_or_bom = has_sig_or_bom
        self._string = decoded_payload
        self._preemptive_declaration = preemptive_declaration
        self._leaves: list[CharsetMatch] = []
        self._unicode_ranges: list[str] | None = None
        self._output_payload: bytes | None = None
        self._output_encoding: str | None = None
        self._fingerprint: int | None = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return iana_name(other) == self.encoding
        return (
            isinstance(other, CharsetMatch)
            and self.encoding == other.encoding
            and self.fingerprint == other.fingerprint
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CharsetMatch):
            raise ValueError
        chaos_difference = abs(self.chaos - other.chaos)
        coherence_difference = abs(self.coherence - other.coherence)
        if chaos_difference < 0.005 and coherence_difference > 0.02:
            return self.coherence > other.coherence
        if chaos_difference < 0.005:
            return self.multi_byte_usage > other.multi_byte_usage
        return self.chaos < other.chaos

    def __str__(self) -> str:
        if self._string is None:
            self._string = bytes(self._payload).decode(self._encoding, "strict")
        return self._string

    def __repr__(self) -> str:
        return f"<CharsetMatch '{self.encoding}' fp({self.fingerprint})>"

    def add_submatch(self, other: CharsetMatch) -> None:
        if not isinstance(other, CharsetMatch) or other == self:
            raise ValueError(f"Unable to add instance <{other.__class__}> as a submatch")
        other._string = None
        self._leaves.append(other)

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def encoding_aliases(self) -> list[str]:
        known: list[str] = []
        for alias, target in aliases.items():
            if self.encoding == alias:
                known.append(target)
            elif self.encoding == target:
                known.append(alias)
        return known

    @property
    def bom(self) -> bool:
        return self._has_sig_or_bom

    @property
    def byte_order_mark(self) -> bool:
        return self._has_sig_or_bom

    @property
    def languages(self) -> list[str]:
        return [language for language, _ in self._languages]

    @property
    def language(self) -> str:
        if self._languages:
            return self._languages[0][0]
        if "ascii" in self.could_be_from_charset:
            return "English"
        return "Unknown"

    @property
    def chaos(self) -> float:
        return self._mean_mess_ratio

    @property
    def coherence(self) -> float:
        return self._languages[0][1] if self._languages else 0.0

    @property
    def percent_chaos(self) -> float:
        return round(self.chaos * 100, 3)

    @property
    def percent_coherence(self) -> float:
        return round(self.coherence * 100, 3)

    @property
    def raw(self) -> bytes | bytearray:
        return self._payload

    @property
    def submatch(self) -> list[CharsetMatch]:
        return self._leaves

    @property
    def has_submatch(self) -> bool:
        return bool(self._leaves)

    @property
    def could_be_from_charset(self) -> list[str]:
        return [self._encoding] + [match.encoding for match in self._leaves]

    @property
    def multi_byte_usage(self) -> float:
        return 0.0 if not self.raw else 1.0 - len(str(self)) / len(self.raw)

    @property
    def alphabets(self) -> list[str]:
        if self._unicode_ranges is None:
            ranges = set()
            for char in str(self):
                if ord(char) < 128:
                    ranges.add("Basic Latin")
                    continue
                try:
                    name = unicodedata.name(char)
                except ValueError:
                    continue
                for key, label in (
                    ("LATIN", "Latin Extended"),
                    ("CYRILLIC", "Cyrillic"),
                    ("GREEK", "Greek and Coptic"),
                    ("ARABIC", "Arabic"),
                    ("HEBREW", "Hebrew"),
                    ("HIRAGANA", "Hiragana"),
                    ("KATAKANA", "Katakana"),
                    ("HANGUL", "Hangul Syllables"),
                    ("CJK", "CJK Unified Ideographs"),
                ):
                    if key in name:
                        ranges.add(label)
                        break
            self._unicode_ranges = sorted(ranges)
        return self._unicode_ranges

    def output(self, encoding: str = "utf_8") -> bytes:
        canonical = iana_name(encoding)
        if self._output_encoding != canonical:
            text = str(self)
            if self._preemptive_declaration and canonical == "utf_8":
                text = re.sub(
                    r"(?i)((?:coding\s*[:=]\s*|charset\s*=\s*[\"']?))[-\w.]+",
                    lambda match: match.group(1) + "utf-8",
                    text,
                    count=1,
                )
            self._output_encoding = canonical
            self._output_payload = text.encode(canonical, "replace")
        return self._output_payload or b""

    @property
    def fingerprint(self) -> int:
        if self._fingerprint is None:
            self._fingerprint = hash(str(self))
        return self._fingerprint


class CharsetMatches:
    def __init__(self, results: list[CharsetMatch] | None = None):
        self._results = sorted(results) if results else []

    def __iter__(self) -> Iterator[CharsetMatch]:
        yield from self._results

    def __getitem__(self, item: int | str) -> CharsetMatch:
        if isinstance(item, int):
            return self._results[item]
        if isinstance(item, str):
            canonical = iana_name(item)
            for result in self._results:
                if canonical in result.could_be_from_charset:
                    return result
        raise KeyError

    def __len__(self) -> int:
        return len(self._results)

    def __bool__(self) -> bool:
        return bool(self._results)

    def append(self, item: CharsetMatch) -> None:
        if not isinstance(item, CharsetMatch):
            raise ValueError(f"Cannot append instance '{item.__class__}' to CharsetMatches")
        for match in self._results:
            if match.fingerprint == item.fingerprint and match.chaos == item.chaos:
                match.add_submatch(item)
                return
        self._results.append(item)
        self._results.sort()

    def best(self) -> CharsetMatch | None:
        return self._results[0] if self._results else None

    def first(self) -> CharsetMatch | None:
        return self.best()
