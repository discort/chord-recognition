import pytest

from chord_recognition.utils import read_audio_from_stream
from chord_recognition.predict import annotate_audio


def test_annotate_audio():
    with open('tests/fixtures/C_Am.mp3', 'rb') as f:
        audio_waveform, Fs = read_audio_from_stream(f)
    result = annotate_audio(audio_waveform, Fs=Fs, nonchord=True)
    expected = [
        (0.0, 2.0434, 'N'),
        (2.0434, 2.1362, 'C'),
        (2.1362, 2.7864, 'N'),
        (2.7864, 2.9722, 'C'),
        (2.9722, 4.9226, 'N'),
        (4.9226, 5.2013, 'A'),
        (5.2013, 6.3158, 'N')
    ]
    assert result == expected
