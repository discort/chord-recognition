import pytest

from chord_recognition.utils import read_audio_from_stream
from chord_recognition.predict import annotate_audio


def test_annotate_audio():
    with open('tests/fixtures/C_Am.mp3', 'rb') as f:
        audio_waveform, Fs = read_audio_from_stream(f)
    result = annotate_audio(audio_waveform, Fs=Fs, nonchord=True)
    expected = [
        (0.0, 1.6718, 'N'),
        (1.6718, 3.3437, 'C'),
        (3.3437, 5.1084, 'N'),
        (5.1084, 6.0372, 'Am')
    ]
    assert result == expected
