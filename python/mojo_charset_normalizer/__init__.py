"""Charset detection and normalization accelerated by Mojo."""

from ._lib import byte_frequencies, entropy, is_valid_utf8
from .api import from_bytes, from_fp, from_path, is_binary
from .legacy import detect
from .models import CharsetMatch, CharsetMatches

VERSION = (0, 1, 0)
__version__ = ".".join(str(part) for part in VERSION)

__all__ = (
    "from_fp", "from_path", "from_bytes", "is_binary", "detect",
    "CharsetMatch", "CharsetMatches", "byte_frequencies", "entropy",
    "is_valid_utf8", "__version__", "VERSION",
)
