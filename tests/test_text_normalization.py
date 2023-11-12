import pytest

from transcription_diff.slice_map import SliceMap
from transcription_diff.text_normalization import normalize_text, expand_abbreviations, standardize_characters


def test_unchanged_text():
    raw_text = "this text is already normalized"
    norm_text, raw2norm = normalize_text(raw_text, "en-us")
    assert raw_text == norm_text
    assert raw2norm == SliceMap.identity(len(raw_text))


def test_edge_cases():
    norm_text, raw2norm = normalize_text("", "en-us")
    assert norm_text == ""
    assert raw2norm == SliceMap.empty()

    norm_text, raw2norm = normalize_text(" ", "en-us")
    assert norm_text == " "
    assert raw2norm == SliceMap.identity(1)

    norm_text, raw2norm = normalize_text(".", "en-us")
    assert norm_text == ""
    assert raw2norm == SliceMap.lerp(1, 0)

    norm_text, raw2norm = normalize_text(" " * 3, "en-us")
    assert norm_text == " "
    assert raw2norm == SliceMap.lerp(3, 1)

    norm_text, raw2norm = normalize_text(". . .", "en-us")
    assert norm_text == " "
    assert raw2norm == SliceMap([slice(0, 0), slice(0, 1), slice(0, 1), slice(0, 1), slice(1, 1)], 1)


def test_abbreviation_expansion():
    norm_text, raw2norm = expand_abbreviations("Hi there dr. House")
    assert norm_text == "Hi there doctor House"
    assert raw2norm == SliceMap.identity(9) + SliceMap.lerp(3, 6) + SliceMap.identity(6)

    norm_text, raw2norm = expand_abbreviations("Hey, jr.! Are you coming jr.?")
    assert norm_text == "Hey, junior! Are you coming junior?"
    assert raw2norm == SliceMap.identity(5) + SliceMap.lerp(3, 6) + SliceMap.identity(17) + \
           SliceMap.lerp(3, 6) + SliceMap.identity(1)

    norm_text, raw2norm = expand_abbreviations("So it goes oct., nov., dec.... Wait, what's after oct.?")
    assert norm_text == "So it goes october, november, december... Wait, what's after october?"
    assert raw2norm == \
           SliceMap.identity(11) + SliceMap.lerp(4, 7) + \
           SliceMap.identity(2) + SliceMap.lerp(4, 8) + \
           SliceMap.identity(2) + SliceMap.lerp(4, 8) + \
           SliceMap.identity(23) + SliceMap.lerp(4, 7) + \
           SliceMap.identity(1)


@pytest.mark.parametrize(
    "text_in, text_out",
    [
        ("Hello world!", "Hello world!"),
        ("é", "é"),
        ("👀", "👀"),

        ("ℍ", "H"),
        ("①", "1"),
        ("¼", "1⁄4"),
    ]
)
def test_character_standardization(text_in: str, text_out: str):
    actual_text_out = "".join(part for part, _ in standardize_characters(text_in))
    assert actual_text_out == text_out
