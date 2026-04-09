import pytest
from docubot import remove_stop_words


def test_removes_common_stop_words():
    assert remove_stop_words("what is the database") == "database"

def test_preserves_content_words():
    assert remove_stop_words("authentication token expired") == "authentication token expired"

def test_mixed_query():
    assert remove_stop_words("how do i reset my password") == "reset password"

def test_case_insensitive():
    assert remove_stop_words("What Is The API") == "api"

def test_empty_string():
    assert remove_stop_words("") == ""

def test_all_stop_words_falls_back_to_original():
    # Query made entirely of stop words should return the original query unchanged
    assert remove_stop_words("what is the") == "what is the"

def test_preserves_word_order():
    result = remove_stop_words("how to install and configure the server")
    assert result == "install configure server"
