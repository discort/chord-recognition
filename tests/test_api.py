import pytest

from chord_recognition.predict import estimate_chords


def test_estimate_chords():
    result = estimate_chords('tests/fixtures/C_Am.mp3', nonchord=True)
    expected = [
        (0.0, 1.6718, 'N'),
        (1.6718, 3.7152, 'C'),
        (3.7152, 4.2725, 'N'),
        (4.2725, 6.0372, 'Am')
    ]
    assert result == expected
