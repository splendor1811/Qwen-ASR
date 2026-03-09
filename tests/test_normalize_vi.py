"""Tests for Vietnamese text normalization."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.normalize_vi import normalize_vietnamese


def test_basic_normalization():
    assert normalize_vietnamese("Xin Chào") == "xin chào"


def test_punctuation_removal():
    assert normalize_vietnamese("Xin chào, tôi là AI!") == "xin chào tôi là ai"


def test_preserves_diacritics():
    text = "Đây là một câu tiếng Việt"
    result = normalize_vietnamese(text)
    assert "đây" in result
    assert "việt" in result


def test_unicode_nfc():
    # Composed vs decomposed forms of Vietnamese characters
    composed = "ă"  # U+0103
    decomposed = "ă"  # a + U+0306
    assert normalize_vietnamese(composed) == normalize_vietnamese(decomposed)


def test_collapse_whitespace():
    assert normalize_vietnamese("xin   chào    bạn") == "xin chào bạn"


def test_empty_string():
    assert normalize_vietnamese("") == ""


def test_numbers_preserved():
    assert normalize_vietnamese("năm 2024") == "năm 2024"


def test_mixed_punctuation():
    text = "Tôi... có 3 con mèo! Bạn thì sao?"
    result = normalize_vietnamese(text)
    assert result == "tôi có 3 con mèo bạn thì sao"


def test_underscore_removal():
    assert normalize_vietnamese("hello_world") == "hello world"


def test_uppercase_vietnamese():
    assert normalize_vietnamese("KHÁCH SẠN NHÀ HÀNG") == "khách sạn nhà hàng"
