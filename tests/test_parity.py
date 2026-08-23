import io

import pytest

import charset_normalizer as upstream
import mojo_charset_normalizer as mojo

ENGLISH = (
    "This is a plain English document with punctuation, numbers 123, and "
    "enough words to identify it reliably.\n"
) * 8
RUSSIAN = (
    "Это русский текст с несколькими предложениями для проверки определения "
    "кодировки. "
) * 8
GREEK = (
    "Αυτό είναι ένα ελληνικό κείμενο για τον έλεγχο της κωδικοποίησης. "
) * 8
JAPANESE = "これは日本語の文章です。文字コードの検出を正しく確認します。" * 8
ARABIC = (
    "هذا نص عربي واضح يحتوي على عدة جمل لاختبار اكتشاف ترميز الأحرف بشكل صحيح. "
) * 8


@pytest.mark.parametrize(
    ("text", "encoding"),
    [
        (ENGLISH, "ascii"),
        (RUSSIAN, "utf-8"),
        (GREEK, "utf-8-sig"),
        (JAPANESE, "utf-16"),
        (RUSSIAN, "utf-32"),
        (GREEK, "utf-16-le"),
        (JAPANESE, "utf-16-be"),
    ],
)
def test_unicode_detection_matches_upstream(text, encoding):
    payload = text.encode(encoding)
    ours = mojo.from_bytes(payload).best()
    theirs = upstream.from_bytes(payload).best()
    assert ours is not None and theirs is not None
    assert ours.encoding == theirs.encoding
    assert str(ours) == str(theirs) == text
    assert ours.bom == theirs.bom
    assert ours.chaos == theirs.chaos == 0.0


@pytest.mark.parametrize(
    ("text", "encoding", "expected"),
    [
        (RUSSIAN, "cp1251", "cp1251"),
        (RUSSIAN, "koi8-r", "koi8_r"),
        (GREEK, "iso8859-7", "cp1253"),
        (JAPANESE, "shift_jis", "cp932"),
        (ARABIC, "cp1256", "cp1256"),
    ],
)
def test_legacy_detection_matches_upstream(text, encoding, expected):
    payload = text.encode(encoding)
    ours = mojo.from_bytes(payload).best()
    theirs = upstream.from_bytes(payload).best()
    assert ours is not None and theirs is not None
    assert ours.encoding == theirs.encoding == expected
    assert str(ours) == str(theirs) == text
    assert abs(ours.chaos - theirs.chaos) < 0.1


def test_euc_jp_normalizes_identically_to_upstream():
    payload = JAPANESE.encode("euc_jp")
    ours = mojo.from_bytes(payload).best()
    theirs = upstream.from_bytes(payload).best()
    assert ours.encoding in {"euc_jp", "euc_jis_2004"}
    assert theirs.encoding in {"euc_jp", "euc_jis_2004"}
    assert str(ours) == str(theirs) == JAPANESE
    assert ours.output() == theirs.output() == JAPANESE.encode()


@pytest.mark.parametrize(
    "payload",
    [
        ENGLISH.encode(),
        b"\x00\x01\x02\xff" * 100,
        bytes(range(256)) * 4,
        b"%PDF-1.7\n" + bytes(range(256)) * 4,
    ],
)
def test_binary_classification_matches_upstream(payload):
    assert mojo.is_binary(payload) is upstream.is_binary(payload)


def test_empty_input_matches_upstream():
    ours = mojo.from_bytes(b"").best()
    theirs = upstream.from_bytes(b"").best()
    assert ours.encoding == theirs.encoding == "utf_8"
    assert str(ours) == str(theirs) == ""
    assert ours.chaos == theirs.chaos == 0.0


def test_large_utf8_sample_path_decodes_lazily():
    text = ("Καλημέρα κόσμε. 日本語 Русский العربية.\n" * 2000)
    match = mojo.from_bytes(text.encode()).best()
    assert match.encoding == "utf_8"
    assert str(match) == text


def test_legacy_sampling_handles_multibyte_window_boundaries():
    text = JAPANESE * 200
    match = mojo.from_bytes(
        text.encode("shift_jis"), steps=3, chunk_size=7
    ).best()
    assert match.encoding == "cp932"
    assert str(match) == text


def test_large_single_byte_match_decodes_lazily():
    text = RUSSIAN * 200
    match = mojo.from_bytes(text.encode("cp1251")).best()
    assert match.encoding == "cp1251"
    assert match._string is None
    assert str(match) == text


def test_detect_legacy_api_matches_upstream():
    assert mojo.detect(ENGLISH.encode()) == upstream.detect(ENGLISH.encode())
    binary = bytes(range(256)) * 4
    assert mojo.detect(binary) == upstream.detect(binary)


def test_from_fp_matches_upstream_without_closing():
    payload = GREEK.encode("utf-8")
    ours_fp = io.BytesIO(payload)
    upstream_fp = io.BytesIO(payload)
    ours = mojo.from_fp(ours_fp).best()
    theirs = upstream.from_fp(upstream_fp).best()
    assert str(ours) == str(theirs) == GREEK
    assert not ours_fp.closed and not upstream_fp.closed


def test_from_path_matches_upstream(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(RUSSIAN.encode("cp1251"))
    ours = mojo.from_path(path).best()
    theirs = upstream.from_path(path).best()
    assert ours.encoding == theirs.encoding == "cp1251"
    assert str(ours) == str(theirs) == RUSSIAN


def test_cp_isolation_and_lookup_by_alias():
    payload = RUSSIAN.encode("cp1251")
    matches = mojo.from_bytes(payload, cp_isolation=["windows-1251"])
    assert matches.best().encoding == "cp1251"
    assert matches["windows-1251"] is matches.best()


def test_cp_exclusion_is_honored():
    payload = RUSSIAN.encode("cp1251")
    matches = mojo.from_bytes(payload, cp_exclusion=["cp1251"])
    with pytest.raises(KeyError):
        matches["cp1251"]


def test_declared_charset_is_prioritized_and_rewritten_on_output():
    text = '<meta charset="windows-1251">' + RUSSIAN
    match = mojo.from_bytes(text.encode("cp1251")).best()
    assert match.encoding == "cp1251"
    normalized = match.output()
    assert b'charset="utf-8"' in normalized
    assert normalized.decode() == '<meta charset="utf-8">' + RUSSIAN


def test_charset_match_properties_match_upstream():
    payload = GREEK.encode("utf-8-sig")
    ours = mojo.from_bytes(payload).best()
    theirs = upstream.from_bytes(payload).best()
    assert ours.encoding == theirs.encoding
    assert ours.byte_order_mark == theirs.byte_order_mark
    assert ours.percent_chaos == theirs.percent_chaos
    assert ours.raw == theirs.raw
    assert ours.output("utf_16") == theirs.output("utf_16")
    assert "Basic Latin" in ours.alphabets


def test_bytearray_and_type_contracts():
    payload = bytearray(JAPANESE.encode())
    assert str(mojo.from_bytes(payload).best()) == JAPANESE
    with pytest.raises(TypeError):
        mojo.from_bytes("not bytes")
    with pytest.raises(TypeError):
        upstream.from_bytes("not bytes")
