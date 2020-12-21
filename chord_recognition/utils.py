import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd

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
        class_ids:   ids of classes to map
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


def compute_chromagram(audio_waveform, Fs, window_size=8192, hop_length=4096,
                       n_chroma=105, norm=None, tuning=0):
    """
    Computes a chromagram from a waveform

    The quarter-tone spectrogram contains only bins corresponding to frequencies
    between Fmin=65Hz (~C2) and Fmax=2100Hz (~C7) and has 24 bins per octave.
    This results in a dimensionality of 105 bins

    Args:
        audio_waveform: Audio time series (np.ndarray [shape=(n,))
        Fs: Sampling rate
        window_size: FFT window size
        hop_length: Hop length
        n_chroma: Number of chroma bins to produce
        norm: Column-wise normalization (if None, no normalization is performed)
        tuning: Deviation from A440 tuning in fractional bins (cents)

    Returns:
        chromagram: Normalized energy for each chroma bin at each frame
                    (np.ndarray [shape=(n_chroma, t)])
    """
    chromagram = librosa.feature.chroma_stft(audio_waveform, sr=Fs, norm=norm,
                                             n_fft=window_size, hop_length=hop_length,
                                             tuning=tuning, n_chroma=n_chroma)
    return chromagram


def log_filtered_spectrogram(audio_waveform, sr, window_size,
                             hop_length, fmin, fmax, num_bands):
    """
    Logarithmic Filtered Spectrogram
    """
    spectrogram = mm.audio.LogarithmicFilteredSpectrogram(
        audio_waveform, sample_rate=sr,
        num_channels=1, frame_size=window_size, hop_size=hop_length,
        fmin=fmin, fmax=fmax, num_bands=num_bands).T
    spectrogram = np.copy(spectrogram)
    return spectrogram


class Rescale:
    """
    Subtracts the frame mean and divides by the standard deviation.
    """

    def __call__(self, frame):
        return preprocess_spectrogram(frame)


def preprocess_spectrogram(spec):
    """
    Subtracts the features mean and divides by the standard deviation.
    """
    return (spec - TRAIN_MEAN) / TRAIN_STD


def scale_data(x, method="norm", axis=0):
    """
    Args:
        method: method of scaling data. Available methods are
            'norm' and 'std' reffering to normalization/standardization
    """
    if method == "norm":
        return normalize(x, axis)
    elif method == "std":
        return standardize(x, axis)
    else:
        raise ValueError(f"method: {method} is not allowed")


def standardize(x, axis=0, eps=1e-20):
    # Rescales data to have a mean of 0 and std of 1
    std = x.std(axis=axis, keepdims=True)
    return (x - x.mean(axis=axis, keepdims=True)) / (std + eps)


def normalize(x, axis=0, eps=1e-20):
    # Rescales data to have values within the range of 0 and 1
    min = x.min(axis=axis, keepdims=True)
    return (x - min) / (x.max(axis=axis, keepdims=True) - min + eps)


def read_audio(path, Fs=None, mono=False, duration=None):
    """Reads an audio file

    Args:
        path: Path to audio file
        Fs: Resample audio to given sampling rate. Use native sampling rate if None.
        mono (bool): Convert multi-channel file to mono.
        duration: only load up to this much audio (in seconds)

    Returns:
        x: Audio time series (np.ndarray [shape=(n,))
        Fs: Sampling rate
    """
    return librosa.load(path, sr=Fs, mono=mono, duration=duration)
