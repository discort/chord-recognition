import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_MEAN = np.array([
    1.1766593, 1.190819, 1.2306786, 1.2365277, 1.2253143,
    1.2072884, 1.2074455, 1.1969299, 1.1865041, 1.16934,
    1.1734127, 1.1627319, 1.1178813, 1.105861, 1.1270006,
    1.1286098, 1.0950696, 1.1379577, 1.154558, 1.1465412,
    1.1189349, 1.0928854, 1.1041877, 1.1422799, 1.152307,
    1.0949671, 1.1060947, 1.1231729, 1.0842474, 1.0400338,
    1.061009, 1.0626515, 1.0920724, 1.0798229, 0.993943,
    0.97623885, 1.0358956, 1.0743601, 1.0072172, 0.9803146,
    0.9873749, 1.0699778, 1.0447679, 1.0051675, 0.9540448,
    0.949218, 0.96742874, 1.0284623, 0.9407551, 0.9143734,
    0.923909, 0.9898914, 0.9162682, 0.88933843, 0.8780559,
    0.9411801, 0.89846206, 0.9526652, 0.88289773, 0.86694413,
    0.8652509, 0.971979, 0.8882464, 0.85740685, 0.8300687,
    0.9459202, 0.87475336, 0.859239, 0.80029184, 0.85274154,
    0.8270254, 0.8876203, 0.7749891, 0.788038, 0.76602864,
    0.8744991, 0.79889584, 0.77842945, 0.7449217, 0.8279752,
    0.76171136, 0.7973064, 0.7300904, 0.7560674, 0.7219486,
    0.81322235, 0.7290665, 0.7380004, 0.70407516, 0.79177314,
    0.7170219, 0.7432798, 0.70134485, 0.7428237, 0.69728315,
    0.75919366, 0.6944593, 0.7118217, 0.67973775, 0.74692565,
    0.67915654, 0.6842601, 0.6526653, 0.7052395, 0.64857495
], dtype=np.float32).reshape(-1, 1)


TRAIN_STD = np.array([
    0.77504474, 0.7684616, 0.77585316, 0.7748402, 0.7592033,
    0.74674773, 0.7330504, 0.72569984, 0.7255572, 0.70355827,
    0.7021016, 0.7126716, 0.6974018, 0.66796744, 0.68983275,
    0.681667, 0.6522506, 0.67407876, 0.6812159, 0.68168485,
    0.67501265, 0.6618095, 0.6595151, 0.6843979, 0.6574317,
    0.6267549, 0.64932907, 0.6738618, 0.63533705, 0.61869276,
    0.6104586, 0.6422936, 0.6065686, 0.6430294, 0.59788215,
    0.5924753, 0.58152324, 0.6374737, 0.57397026, 0.5920942,
    0.5711934, 0.61864525, 0.5734692, 0.60801095, 0.5611059,
    0.5776549, 0.55234265, 0.6100096, 0.54523194, 0.5645996,
    0.53230417, 0.58506256, 0.52854645, 0.5606829, 0.52815986,
    0.5733189, 0.5276037, 0.5814292, 0.52713394, 0.5479652,
    0.5203032, 0.583182, 0.5254504, 0.5453514, 0.5133796,
    0.56340516, 0.51529235, 0.5535773, 0.5052468, 0.5391684,
    0.496269, 0.5445258, 0.48395267, 0.50321585, 0.4791966,
    0.5371042, 0.48392695, 0.5140148, 0.47538552, 0.50693053,
    0.4641765, 0.50127834, 0.45626664, 0.47700867, 0.450689,
    0.4972031, 0.44734702, 0.46943438, 0.4393932, 0.47841448,
    0.43976146, 0.47201458, 0.44033337, 0.46055573, 0.43509123,
    0.46888384, 0.431725, 0.44620052, 0.42624134, 0.45570937,
    0.4187576, 0.43385547, 0.41336793, 0.43448588, 0.40724447
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
