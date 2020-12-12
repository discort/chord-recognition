import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_MEAN = np.array([
    1.0545825, 1.1157063, 1.155371, 1.1603547, 1.1729534,
    1.2336454, 1.2762042, 1.2693729, 1.235095, 1.1883477,
    1.1782604, 1.1558412, 1.1203424, 1.1404415, 1.1638157,
    1.1424468, 1.1209729, 1.1820542, 1.1855319, 1.1148462,
    1.0918729, 1.0933912, 1.1294774, 1.1558713, 1.1645583,
    1.1344076, 1.1639677, 1.1817598, 1.112456, 1.0520899,
    1.0882324, 1.0977433, 1.0734009, 1.0455191, 1.0147096,
    1.02442, 1.0692517, 1.0844218, 1.0069767, 0.96993095,
    1.0347502, 1.1361623, 1.0539485, 0.9686497, 0.98853296,
    1.0054287, 0.9924657, 1.0117942, 0.97333753, 0.95913273,
    0.99437684, 1.0425543, 0.9354105, 0.87906283, 0.9253904,
    0.9792612, 0.9110867, 0.9079047, 0.8982375, 0.89466214,
    0.89902437, 0.94244546, 0.8706192, 0.83199465, 0.86670256,
    0.9542005, 0.8647041, 0.80323344, 0.82579154, 0.8727365,
    0.8436244, 0.8525542, 0.79913485, 0.79997796, 0.7960668,
    0.85846835, 0.7807801, 0.7299946, 0.7565614, 0.8312265,
    0.7610649, 0.7341334, 0.73727995, 0.75366116, 0.7380677,
    0.77720743, 0.7252766, 0.7055482, 0.7165931, 0.77815366,
    0.7275544, 0.7058122, 0.71843517, 0.74853796, 0.71332955,
    0.7265763, 0.71616197, 0.7275288, 0.7200941, 0.75017625,
    0.70057285, 0.6834786, 0.6802684, 0.7136173, 0.65647936],
    dtype=np.float32).reshape(-1, 1)


TRAIN_STD = np.array([
    0.57545036, 0.5945545, 0.60891676, 0.6039973, 0.5977075,
    0.6026711, 0.59741044, 0.62332666, 0.6095669, 0.5766476,
    0.58590144, 0.57777613, 0.56965715, 0.56573904, 0.57898074,
    0.5511234, 0.52893454, 0.5689488, 0.5648477, 0.5260298,
    0.5286965, 0.54346913, 0.5434677, 0.5529419, 0.5115087,
    0.50015783, 0.56050825, 0.56666285, 0.5239892, 0.48211408,
    0.4975923, 0.52994716, 0.47885782, 0.4883289, 0.49553183,
    0.48549977, 0.47653356, 0.5123332, 0.4700112, 0.4671782,
    0.48314932, 0.51432, 0.4620244, 0.4551256, 0.4572728,
    0.47547683, 0.46080846, 0.47487983, 0.44366184, 0.45892397,
    0.45803463, 0.48088467, 0.4316556, 0.42992553, 0.44936305,
    0.47128198, 0.43781966, 0.44359058, 0.44012642, 0.45353717,
    0.45300913, 0.47089598, 0.4236528, 0.42860493, 0.45397565,
    0.47356617, 0.41779813, 0.41290593, 0.4342716, 0.45906097,
    0.41810426, 0.42652887, 0.40596431, 0.41523975, 0.41423038,
    0.4367769, 0.39469868, 0.38992113, 0.4042664, 0.4275456,
    0.3873884, 0.3850605, 0.38483405, 0.39344853, 0.38787997,
    0.40294746, 0.37361476, 0.37624675, 0.37987715, 0.39796752,
    0.37025842, 0.36155415, 0.37276706, 0.38580754, 0.36882412,
    0.37193057, 0.3688181, 0.3742507, 0.36591148, 0.37508374,
    0.3583005, 0.35146096, 0.34786066, 0.35609233, 0.33904684],
    dtype=np.float32).reshape(-1, 1)


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
