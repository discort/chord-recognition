import pytest

from chord_recognition.dataset import MirexFameDataset


def test_dataset():
    dataset = MirexFameDataset(audio_dir="tests/fixtures/mp3/",
                               ann_dir="tests/fixtures/annotations/",
                               window_size=8192, hop_length=4096,
                               context_size=7)
    assert len(dataset) == 42
    input_, label = dataset[0]
    assert input_.shape == (1, 105, 15)
    assert label.shape == (25,)
