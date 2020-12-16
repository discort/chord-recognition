import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_MEAN = np.array([
	1.0433939 , 1.0977385 , 1.134623  , 1.1378937 , 1.1512516 ,
       1.2113197 , 1.2566348 , 1.2560139 , 1.2222104 , 1.171966  ,
       1.1651376 , 1.1466012 , 1.1103952 , 1.1268272 , 1.1500771 ,
       1.1299808 , 1.1038449 , 1.1627611 , 1.1700675 , 1.101504  ,
       1.0792125 , 1.0813347 , 1.1179888 , 1.1454918 , 1.1511924 ,
       1.1184406 , 1.1500229 , 1.1726462 , 1.1057829 , 1.0444188 ,
       1.0751398 , 1.087159  , 1.0756793 , 1.0528    , 1.0134451 ,
       1.0211025 , 1.0674847 , 1.0855371 , 1.0068258 , 0.9665336 ,
       1.0316452 , 1.1360065 , 1.0571939 , 0.96906334, 0.98940736,
       1.0089035 , 0.99880975, 1.0185994 , 0.97235113, 0.95326465,
       0.9926792 , 1.0431027 , 0.9396895 , 0.87825817, 0.9196811 ,
       0.97292835, 0.9147039 , 0.9167589 , 0.9026672 , 0.89480615,
       0.89939755, 0.94569135, 0.8745289 , 0.829581  , 0.8657507 ,
       0.95354575, 0.8690921 , 0.8065337 , 0.82976377, 0.87309927,
       0.8477255 , 0.85919833, 0.799786  , 0.79554707, 0.7976942 ,
       0.86156374, 0.79137105, 0.7308446 , 0.7555842 , 0.82768416,
       0.76486385, 0.74102044, 0.73914295, 0.75352913, 0.739315  ,
       0.7804631 , 0.7259804 , 0.6995562 , 0.7179888 , 0.7800916 ,
       0.73008204, 0.70446837, 0.7189802 , 0.74699485, 0.71367794,
       0.7282223 , 0.7147906 , 0.72310764, 0.72080696, 0.7510512 ,
       0.7084636 , 0.6843683 , 0.6789441 , 0.70988786, 0.65600324
], dtype=np.float32).reshape(-1, 1)


TRAIN_STD = np.array([
   0.5751935 , 0.5904691 , 0.6025101 , 0.5993458 , 0.5950813 ,
       0.5990092 , 0.5961791 , 0.6232101 , 0.6101956 , 0.5767615 ,
       0.5871233 , 0.58427507, 0.57403946, 0.5642899 , 0.5751121 ,
       0.54718935, 0.5267585 , 0.56561166, 0.5626441 , 0.5262261 ,
       0.52840936, 0.54254776, 0.54436535, 0.55303186, 0.51102495,
       0.50023293, 0.55871123, 0.56348044, 0.5241162 , 0.4835659 ,
       0.49731904, 0.5276608 , 0.47929865, 0.4904505 , 0.49517938,
       0.4844203 , 0.4753199 , 0.5107833 , 0.4726827 , 0.46897042,
       0.48395953, 0.51323223, 0.46445793, 0.45616883, 0.4590285 ,
       0.4757267 , 0.45986256, 0.47522125, 0.44530892, 0.45969093,
       0.45764342, 0.47962052, 0.4356946 , 0.43084928, 0.44871774,
       0.46923456, 0.4365071 , 0.44435462, 0.44021037, 0.45194814,
       0.45143878, 0.46993887, 0.42543238, 0.42726406, 0.4521723 ,
       0.47090268, 0.42252257, 0.41602376, 0.43579134, 0.45792297,
       0.41641313, 0.4265878 , 0.40689188, 0.41323692, 0.41317746,
       0.43537238, 0.40033016, 0.38946396, 0.40227497, 0.42193437,
       0.38550532, 0.3855158 , 0.38613334, 0.39218318, 0.3875235 ,
       0.4029321 , 0.37575778, 0.37237975, 0.3788224 , 0.39682448,
       0.37182742, 0.36162904, 0.37304842, 0.3849293 , 0.3695364 ,
       0.37293512, 0.36870933, 0.3719063 , 0.366115  , 0.37635323,
       0.36149505, 0.35047954, 0.34590316, 0.35352564, 0.33845058
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
