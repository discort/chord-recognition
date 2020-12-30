import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

TRAIN_MEAN = np.array([
    1.1535486, 1.1963599, 1.2407198, 1.2483087, 1.248103,
    1.2675027, 1.2818235, 1.2709185, 1.2528844, 1.2297423,
    1.2248168, 1.2064269, 1.166605, 1.165775, 1.1843067,
    1.1813436, 1.1562898, 1.1976565, 1.2061249, 1.1824946,
    1.1667169, 1.1485462, 1.1610315, 1.1896667, 1.2048994,
    1.1618224, 1.172531, 1.1882926, 1.1438125, 1.1026299,
    1.1245255, 1.1173555, 1.1248295, 1.113029, 1.0532343,
    1.0408038, 1.086648, 1.1151811, 1.0518891, 1.0252558,
    1.0496762, 1.1288368, 1.0843059, 1.0383109, 1.0172894,
    1.0118431, 1.0071429, 1.0517013, 0.995275, 0.9760286,
    0.98881334, 1.0408012, 0.96615016, 0.93717265, 0.9425446,
    0.9875599, 0.9337886, 0.97151256, 0.93218154, 0.91304576,
    0.90731174, 0.9854886, 0.9153977, 0.8863871, 0.87839717,
    0.96981126, 0.89955974, 0.8766584, 0.8514744, 0.88902956,
    0.85883224, 0.8987339, 0.8194018, 0.82238203, 0.8045283,
    0.88917154, 0.8202385, 0.7963995, 0.783093, 0.85008156,
    0.78606737, 0.8020942, 0.7677416, 0.7802751, 0.7495019,
    0.81911945, 0.7557811, 0.75558305, 0.7351936, 0.8049089,
    0.7467735, 0.7566471, 0.7381206, 0.770315, 0.7267599,
    0.7687228, 0.73058313, 0.74129087, 0.719047, 0.77024543,
    0.71390146, 0.71256953, 0.68735486, 0.72739595, 0.6727527
], dtype=np.float32).reshape(-1, 1)


TRAIN_STD = np.array([
    0.6639351, 0.66715026, 0.6714438, 0.66695094, 0.65177083,
    0.6411022, 0.6301112, 0.63807756, 0.6352697, 0.6182556,
    0.6196286, 0.61717576, 0.6054105, 0.58756673, 0.6057221,
    0.59059525, 0.56438607, 0.58822954, 0.58928454, 0.57774156,
    0.579073, 0.5789587, 0.5722514, 0.586899, 0.5504802,
    0.533124, 0.57190263, 0.58796775, 0.54912907, 0.52810067,
    0.5278655, 0.56118923, 0.5115619, 0.5388198, 0.5204856,
    0.5126983, 0.49768034, 0.5463397, 0.49518842, 0.5109048,
    0.5006761, 0.5331656, 0.48509595, 0.50956833, 0.48347816,
    0.5028641, 0.47609133, 0.5161718, 0.4701013, 0.48792738,
    0.4678328, 0.5048907, 0.45573363, 0.4807731, 0.46838856,
    0.50003296, 0.45564762, 0.48540917, 0.46058327, 0.47955713,
    0.46484232, 0.5031166, 0.4536096, 0.47329658, 0.46379933,
    0.49260947, 0.44385004, 0.46998852, 0.4547681, 0.4808286,
    0.4350567, 0.46287525, 0.42643744, 0.44054255, 0.42733952,
    0.46425927, 0.42057785, 0.44329247, 0.42263454, 0.44508255,
    0.40637848, 0.42766428, 0.40250552, 0.41755566, 0.40044418,
    0.4309611, 0.39276433, 0.4099779, 0.3925973, 0.41702327,
    0.38635302, 0.4022788, 0.38901472, 0.40674937, 0.3826903,
    0.40297672, 0.3815937, 0.39096436, 0.3769787, 0.39496577,
    0.36949632, 0.3775483, 0.3618002, 0.37719214, 0.3563225
], dtype=np.float32).reshape(-1, 1)


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


def context_window(x, context_size):
    """
    Iterate through each item of sequence padded with elements (with context)

    math::
        X_{i} = [l_{i-C}, ..., l_{i}, ..., l_{i+C}]
        i - index of the target
        C - context size

    Args:
        x: ndarray (M, N) where N - length
        context_size: amount of elements padded to each item
    """
    assert context_size % 2 == 1, "context_size must be odd"
    assert x.shape[1] > 2 * context_size + 1, "Low size x"
    assert x.ndim == 2, "X dimension must be 2"

    pad_number = context_size
    left_pad_idx = pad_number
    N = x.shape[1]
    dtype = x.dtype
    right_pad_idx = 1
    for i in range(N):
        if i - pad_number < 0:
            right = x[:, range(0, i + pad_number + 1)]
            left_pad = x[:, :left_pad_idx]
            window = np.concatenate([left_pad, right], axis=1)
            left_pad_idx -= 1
        elif i + pad_number >= N:
            left = x[:, range(i - pad_number, N)]
            right_pad = x[:, -right_pad_idx:]
            window = np.concatenate([left, right_pad], axis=1)
            right_pad_idx += 1
        else:
            indexes = list(range(i - pad_number, i)) + list(range(i, i + pad_number + 1))
            window = x[:, indexes]
        yield window.astype(dtype)


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


def log_compression(v, gamma=1):
    """Logarithmically compresses a value or array

    Args:
        v: Value or array
        gamma: Compression factor

    Returns:
        v_compressed: Compressed value or array
    """
    return np.log(1 + gamma * v)


def log_filtered_spectrogram(audio_waveform, sr, window_size,
                             hop_length, fmin, fmax, num_bands):
    """
    Args:
        audio_waveform: audio time series (np.ndarray [shape=(n,))
        sr: audio sampling rate of `audio_waveform`
        window_size: FFT window size
        hop_length: Hop length for STFT

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


class Rescale:
    """
    Subtracts the frame mean and divides by the standard deviation.
    """

    def __call__(self, frame):
        return standardize(frame, TRAIN_MEAN, TRAIN_STD)


def standardize(x, mean, std, eps=1e-20):
    """
    Rescale inputs to have a mean of 0 and std of 1
    """
    return (x - mean) / (std + eps)


def destandardize(x, mean, std):
    """Undo preprocessing on a frame"""
    return x * std + mean


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
