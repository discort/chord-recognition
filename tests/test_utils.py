import pytest

import numpy as np

from chord_recognition import utils

np.random.seed(0)


def test_expand_labels():
    x = [(100, [25]),
         (100, []),
         (100, [10, 1]),
         (43, [25])]
    result = utils.expand_labels(x)
    assert len(result) == 343
    assert result[:100] == [25 for _ in range(100)]
    assert result[100:200] == [0 for _ in range(100)]
    assert result[200:250] == [10 for _ in range(50)]
    assert result[250:300] == [1 for _ in range(50)]
    assert result[300:] == [25 for _ in range(43)]


def test_batch_sequence():
    x = np.random.randn(1543, 105)
    result = utils.batch_sequence(x, 100, last=True)
    assert len(result) == 16
    expected = [
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (100, 105),
        (43, 105)]
    assert [s.shape for s in result] == expected
