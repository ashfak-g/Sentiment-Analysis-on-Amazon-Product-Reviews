"""
Unit tests for text preprocessing module.
"""

import pytest
from src.text_preprocessor import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor(remove_stopwords=True, lemmatize=True)

def test_clean_text_basic(preprocessor):
    raw = "This is a VERY GOOD app! <br> Loved it 100%."
    cleaned = preprocessor.clean_text(raw)
    assert "good" in cleaned
    assert "app" in cleaned
    assert "100" not in cleaned
    assert "<br>" not in cleaned

def test_negation_preservation(preprocessor):
    raw = "This product is not bad at all."
    cleaned = preprocessor.clean_text(raw)
    assert "not" in cleaned

def test_empty_string(preprocessor):
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""
