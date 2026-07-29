from __future__ import annotations

import warnings

from .api import from_bytes


def detect(
    byte_str: bytes, should_rename_legacy: bool = False, **kwargs
) -> dict[str, str | float | None]:
    if kwargs:
        warnings.warn(
            f"charset-normalizer disregard arguments '{','.join(kwargs)}' in legacy function detect()",
            stacklevel=2,
        )
    if not isinstance(byte_str, (bytearray, bytes)):
        raise TypeError(f"Expected object of type bytes or bytearray, got: {type(byte_str)}")
    result = from_bytes(byte_str).best()
    if result is None:
        return {"encoding": None, "language": "", "confidence": None}
    encoding = result.encoding
    if encoding == "utf_8" and result.bom:
        encoding = "utf_8_sig"
    return {
        "encoding": encoding,
        "language": "" if result.language == "Unknown" else result.language,
        "confidence": 1.0 - result.chaos,
    }
