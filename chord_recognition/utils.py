import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_MEAN = np.array([
    1.1384697, 1.172602, 1.2092729, 1.2085352, 1.2114284,
    1.2323847, 1.2524031, 1.239282, 1.2179618, 1.1861023,
    1.1860715, 1.1716574, 1.1286707, 1.1361063, 1.1670705,
    1.1563107, 1.1181066, 1.1688118, 1.180461, 1.142441,
    1.1148702, 1.1007199, 1.126893, 1.1702741, 1.1769226,
    1.1216727, 1.1457078, 1.1660001, 1.1083533, 1.0497278,
    1.0802087, 1.0877109, 1.0918213, 1.0731918, 1.0091032,
    0.99810565, 1.0607874, 1.100123, 1.0159003, 0.9758182,
    1.0129449, 1.1086781, 1.0508573, 0.9879717, 0.9711812,
    0.9776736, 0.98673916, 1.0397363, 0.9636918, 0.9311401,
    0.9599091, 1.0285356, 0.9306442, 0.8853129, 0.90602046,
    0.9678137, 0.9072222, 0.9367507, 0.8951169, 0.8805168,
    0.8878696, 0.97912675, 0.88912004, 0.84396195, 0.8505009,
    0.96198606, 0.8764313, 0.8378944, 0.82021296, 0.8738495,
    0.84405583, 0.8894078, 0.79763675, 0.79638517, 0.785473,
    0.8849061, 0.80110455, 0.7604083, 0.7592191, 0.8441419,
    0.77028775, 0.7785264, 0.7418176, 0.7609413, 0.736937,
    0.81540257, 0.7391286, 0.72907734, 0.71812826, 0.80134356,
    0.73025954, 0.73154205, 0.7167186, 0.7573429, 0.7128965,
    0.7583348, 0.7157476, 0.725329, 0.7051278, 0.7640732,
    0.6992757, 0.69098455, 0.67340344, 0.72268534, 0.6629429
], dtype=np.float32).reshape(-1, 1)


TRAIN_STD = np.array([
    0.7020633, 0.70347583, 0.7093843, 0.70429134, 0.6896269,
    0.68639386, 0.67546177, 0.6815982, 0.6751033, 0.6474338,
    0.65186954, 0.6571184, 0.6432216, 0.6227175, 0.649846,
    0.63159025, 0.5974112, 0.6281935, 0.63074076, 0.6155229,
    0.61358327, 0.6096751, 0.60859853, 0.6332205, 0.5987685,
    0.57136786, 0.61292005, 0.6321965, 0.58866096, 0.5594436,
    0.5617029, 0.59441626, 0.5509931, 0.5800363, 0.5537678,
    0.5440129, 0.53714067, 0.59071904, 0.53036034, 0.53702253,
    0.53215307, 0.5744936, 0.5232208, 0.5418799, 0.51370275,
    0.5324249, 0.51186395, 0.5581139, 0.50340325, 0.5162874,
    0.4998691, 0.5429975, 0.48545352, 0.50369316, 0.49381208,
    0.52941316, 0.48571104, 0.52230823, 0.49020427, 0.5063388,
    0.49165893, 0.5399845, 0.48279455, 0.49333042, 0.48795664,
    0.52540356, 0.47368988, 0.49547786, 0.47697413, 0.5055027,
    0.4631885, 0.49760398, 0.45280775, 0.46500167, 0.4527659,
    0.49887225, 0.4475827, 0.46163163, 0.4447253, 0.47409704,
    0.43165362, 0.4550979, 0.42689267, 0.44241858, 0.42506352,
    0.46075764, 0.4176474, 0.43076017, 0.41614673, 0.44682375,
    0.41124922, 0.42629284, 0.41318375, 0.4317994, 0.40759405,
    0.4313486, 0.4073104, 0.416611, 0.4020821, 0.42415863,
    0.39432144, 0.40006495, 0.38701987, 0.4037515, 0.38083082
], dtype=np.float32).reshape(-1, 1)


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
