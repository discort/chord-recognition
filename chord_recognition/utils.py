import itertools
import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")


def ctc_greedy_decoder(logits, blank_label=0, merge_repeated=True):
    """
    Performs greedy decoding on the logits given in input (best path)
    Args:
        scores np.array[max_time, batch_size, num_classes] - input logits
        blank_index (int) - blank index number
        merge_repeated (bool) - if consecutive logits' maximum indices are the same,
            only the first of these is emitted
    Returns:
        list[list] - The vector stores the decoded classes.
    """
    max_scores = np.argmax(logits, 2).T
    decodes = []
    for batch in max_scores:
        decode = []
        for j, index in enumerate(batch):
            if index != blank_label:
                if merge_repeated and j != 0 and index == batch[j-1]:
                    continue
                decode.append(index.item())
        decodes.append(decode)
    return decodes


def one_hot(class_ids, num_classes):
    """
    Create one-hot encoding of class ids

    Args:
        class_ids: ids of classes to map
        num_classes: number of classes

    Returns:
        one-hot encoding of class ids
    """
    oh = np.zeros((len(class_ids), num_classes), dtype=np.int64)
    oh[np.arange(len(class_ids)), class_ids] = 1

    # make sure one-hot encoding corresponds to class ids
    assert (oh.argmax(axis=1) == class_ids).all()
    # make sure there is only one id set per vector
    assert (oh.sum(axis=1) == 1).all()

    return oh


def split_with_context(x, context_size, pad_data=None):
    """
    Iterate through each item of sequence padded with elements (with context)

    math::
        X_{i} = [l_{i-C}, ..., l_{i}, ..., l_{i+C}]
        i - index of the target
        C - context size

    Args:
        x: np.ndarray[shape=(n_features, n_frames)]
        context_size: amount of elements padded to each item
    """
    assert context_size % 2 == 1, "context_size must be odd"
    assert x.ndim == 2, "X dimension must be 2"
    n_features, n_frames = x.shape
    dtype = x.dtype

    if not pad_data:
        pad_data = np.zeros((n_features, context_size))
    padded = np.hstack([pad_data, x, pad_data])

    start = context_size
    stop = padded.shape[1] - context_size
    for i in range(start, stop):
        indexes = list(range(i - context_size, i)) + list(range(i, i + context_size + 1))
        window = padded[:, indexes]
        yield window.astype(dtype)


def expand_labels(inputs, blank_label=0):
    """
    Expand given list of labels into sequence seized by seq_len

    Ex.
    >> inputs
    >> [(100, [25]),
        (100, []),
        (100, [10, 1]),
        (43, [25])]

    Args:
        inputs [(seq_len, [list_of_labels])]
    """
    result = []
    for i, (seq_len, labels) in enumerate(inputs):
        label_rate = 1. / len(labels) if len(labels) > 0 else 0
        labels_stack = []
        for item in labels:
            elems = itertools.repeat(item, int(seq_len * label_rate))
            labels_stack.extend(list(elems))

        # Handle blank labels list
        if not labels_stack:
            labels_stack = list(itertools.repeat(blank_label, seq_len))

        result.extend(labels_stack)
    return result


def batch_sequence(sequence, batch_size, last=False):
    result = []
    stack = []
    for item in sequence:
        if len(stack) < batch_size:
            stack.append(item)
        if len(stack) == batch_size:
            result.append(np.stack(stack))
            stack = []
    if last and len(stack) > 0:
        result.append(np.stack(stack))
    return result


def read_csv(fn, header=False, add_label=False, sep=' '):
    """Reads a CSV file

    Args:
        fn: Filename
        header: Boolean
        add_label: Add column with constant value of `add_label`

    Returns:
        df: Pandas DataFrame
    """
    df = pd.read_csv(fn, sep=sep, keep_default_na=False, header=0 if header else None)
    if add_label:
        assert 'label' not in df.columns, 'Label column must not exist if `add_label` is True'
        df = df.assign(label=[add_label] * len(df.index))
    return df


def exponential_smoothing(series, alpha):
    """given a series and alpha, return series of expoentially smoothed points"""
    results = np.zeros_like(series)

    # first value remains the same as series,
    # as there is no history to learn from
    results[0] = series[0]
    for t in range(1, series.shape[0]):
        results[t] = alpha * series[t] + (1 - alpha) * results[t - 1]

    return results


def log_filtered_spectrogram(audio_waveform, sr, window_size,
                             hop_length, fmin, fmax, num_bands):
    """
    Args:
        audio_waveform: audio time series (np.ndarray [shape=(n,))
        sr: audio sampling rate of `audio_waveform`
        window_size: FFT window size
        hop_length: Hop length for STFT

    Returns:
        log-filtered specgrogram: np.ndarray[shape=(n_features, n_frames)]

    """
    spectrogram = mm.audio.LogarithmicFilteredSpectrogram(
        audio_waveform, sample_rate=sr,
        num_channels=1, frame_size=window_size, hop_size=hop_length,
        fmin=fmin, fmax=fmax, num_bands=num_bands).T
    spectrogram = np.copy(spectrogram)
    return spectrogram


def batch_exponential_smoothing(x, alpha):
    batsches, _ = x.shape
    result = []
    for i in range(batsches):
        x_smooth = exponential_smoothing(x[i, :].numpy(), alpha)
        result.append(torch.from_numpy(x_smooth))
    return torch.stack(result)


def read_audio(path, sr=None, mono=True, duration=None):
    """Load an audio file as a floating point time series.

    Args:
        path: path to an audio file
        sr: target sampling rate
        mono (bool): Convert multi-channel file to mono.
        duration: only load up to this much audio (in seconds)

    Returns:
        y: Audio time series (np.ndarray [shape=(n,))
        sr: Sampling rate of y
    """
    return librosa.load(path, sr=sr, mono=mono, duration=duration)
