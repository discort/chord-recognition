import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRAIN_MEAN = np.array([
	-0.01944327, -0.03022057, -0.03407357, -0.0371877 , -0.03630774,
       -0.03704427, -0.03275802, -0.02143209, -0.02113691, -0.02840916,
       -0.02239726, -0.01599289, -0.01746228, -0.02406553, -0.02372977,
       -0.02261954, -0.03238214, -0.03391011, -0.02737833, -0.02536512,
       -0.02394619, -0.02218445, -0.02113966, -0.0187707 , -0.02612999,
       -0.03192439, -0.02487764, -0.01608339, -0.01273426, -0.01591197,
       -0.02631242, -0.01997296,  0.00475905,  0.01491057, -0.00255164,
       -0.00683252, -0.00370804,  0.0021774 , -0.00032139, -0.00727319,
       -0.00642663, -0.00030318,  0.00702473,  0.00090853,  0.00191134,
        0.00730889,  0.01376735,  0.01433014, -0.00222297, -0.01278611,
       -0.0037065 ,  0.00114114,  0.00991296, -0.00187157, -0.0127054 ,
       -0.01343762,  0.00826215,  0.01995925,  0.01006451,  0.00031818,
        0.00082352,  0.00689332,  0.00922772, -0.00563151, -0.00209631,
       -0.00138267,  0.01050208,  0.00799367,  0.00914644,  0.00078994,
        0.00980817,  0.01557684,  0.00160388, -0.01067097,  0.00392894,
        0.00708615,  0.02683305,  0.00218001, -0.00241695, -0.0082857 ,
        0.00980626,  0.01788546,  0.00484114, -0.00033484,  0.00321431,
        0.00807969,  0.00188366, -0.01592573,  0.00367483,  0.00487   ,
        0.00682664, -0.00371638,  0.0014623 , -0.00399923,  0.00094416,
        0.00442575, -0.00371757, -0.01181384,  0.00194804,  0.00233346,
        0.02202229,  0.00253142, -0.00380758, -0.01047348, -0.00140413
], dtype=np.float32).reshape(-1, 1)


TRAIN_STD = np.array([
    0.9995538 , 0.9931289 , 0.9894786 , 0.99229866, 0.9956061 ,
       0.99392396, 0.99793893, 0.9998128 , 1.0010314 , 1.0001974 ,
       1.002085  , 1.011248  , 1.0076929 , 0.9974382 , 0.9933181 ,
       0.9928616 , 0.9958858 , 0.99413437, 0.9960988 , 1.000373  ,
       0.999457  , 0.99830496, 1.0016515 , 1.0001627 , 0.9990543 ,
       1.00015   , 0.9967942 , 0.9943842 , 1.0002426 , 1.0030115 ,
       0.9994509 , 0.9956856 , 1.0009205 , 1.0043447 , 0.9992885 ,
       0.9977763 , 0.9974534 , 0.996975  , 1.0056841 , 1.0038365 ,
       1.0016768 , 0.9978852 , 1.0052671 , 1.0022923 , 1.0038393 ,
       1.0005255 , 0.99794716, 1.0007187 , 1.0037129 , 1.0016718 ,
       0.9991453 , 0.9973712 , 1.009357  , 1.0021485 , 0.99856335,
       0.9956558 , 0.99700207, 1.0017222 , 1.0001906 , 0.99649614,
       0.99653363, 0.99796766, 1.0042005 , 0.99687153, 0.9960279 ,
       0.9943761 , 1.0113078 , 1.0075508 , 1.0034992 , 0.99752116,
       0.99595517, 1.000138  , 1.0022849 , 0.99517673, 0.9974583 ,
       0.99678427, 1.0142677 , 0.99882734, 0.9950742 , 0.98687524,
       0.9951389 , 1.0011826 , 1.003376  , 0.99678403, 0.9990811 ,
       0.9999618 , 1.005736  , 0.98972213, 0.99722326, 0.9971278 ,
       1.0042378 , 1.0002072 , 1.0007553 , 0.99772346, 1.0019313 ,
       1.0027008 , 0.9997047 , 0.99373597, 1.0005562 , 1.003384  ,
       1.0089161 , 0.9972078 , 0.99437267, 0.99279165, 0.99824136
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
