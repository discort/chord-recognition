import warnings

import librosa
import madmom as mm
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

TRAIN_MEAN = np.array([1.1528504 , 1.1952968 , 1.2392036 , 1.2462871 , 1.24655   ,
       1.266298  , 1.281114  , 1.2699821 , 1.2515444 , 1.2269336 ,
       1.2234626 , 1.2057283 , 1.1658561 , 1.1653907 , 1.1851586 ,
       1.182024  , 1.1542513 , 1.1970143 , 1.2064042 , 1.181029  ,
       1.1649486 , 1.1470466 , 1.1601826 , 1.1891675 , 1.203288  ,
       1.1597621 , 1.1717209 , 1.1889725 , 1.1430693 , 1.0995636 ,
       1.1220607 , 1.116234  , 1.1237054 , 1.1114032 , 1.0513473 ,
       1.0387721 , 1.0854193 , 1.1150621 , 1.0501912 , 1.0221803 ,
       1.0477622 , 1.1282184 , 1.0827678 , 1.0352527 , 1.0153502 ,
       1.011273  , 1.0070807 , 1.0515583 , 0.99370605, 0.9734555 ,
       0.98849654, 1.042048  , 0.9652285 , 0.93440753, 0.9408287 ,
       0.9871822 , 0.9332332 , 0.9702542 , 0.93083674, 0.9117122 ,
       0.90676874, 0.9852673 , 0.914235  , 0.88369286, 0.87745064,
       0.97001785, 0.8985701 , 0.87393695, 0.8503802 , 0.88912153,
       0.8590631 , 0.89842707, 0.8186334 , 0.82091594, 0.804582  ,
       0.89018047, 0.8200434 , 0.79411435, 0.78202397, 0.84995896,
       0.78557074, 0.8002956 , 0.7665309 , 0.7796598 , 0.7495301 ,
       0.81901085, 0.755604  , 0.7539985 , 0.7348397 , 0.8055555 ,
       0.7463677 , 0.7548251 , 0.73735017, 0.770473  , 0.7269362 ,
       0.7686779 , 0.73089814, 0.7411712 , 0.7194262 , 0.770539  ,
       0.7141707 , 0.7118278 , 0.6871908 , 0.72763443, 0.6730587 ],
dtype=np.float32).reshape(-1, 1)

TRAIN_STD = np.array([0.6644172 , 0.6671606 , 0.6716639 , 0.6677901 , 0.6526045 ,
       0.64249784, 0.63171124, 0.63970405, 0.6359139 , 0.61846435,
       0.61986953, 0.61814475, 0.6067679 , 0.5885088 , 0.6073641 ,
       0.59114575, 0.56461436, 0.58913094, 0.5899034 , 0.578506  ,
       0.5795537 , 0.57939726, 0.5733028 , 0.5879359 , 0.55203944,
       0.53380144, 0.5733533 , 0.5887266 , 0.5500897 , 0.52844423,
       0.5284046 , 0.56190485, 0.5124435 , 0.5396747 , 0.5208862 ,
       0.5127261 , 0.49886632, 0.54786074, 0.49644312, 0.51090163,
       0.5010824 , 0.53383946, 0.48612386, 0.5102481 , 0.48435327,
       0.50354016, 0.47737014, 0.51724416, 0.47075605, 0.4880118 ,
       0.46877486, 0.50603914, 0.4566514 , 0.48071814, 0.46920437,
       0.5006895 , 0.45639306, 0.4864319 , 0.4615794 , 0.48001572,
       0.46543333, 0.50390124, 0.45437205, 0.4728472 , 0.46429577,
       0.49306855, 0.44457808, 0.46984082, 0.4550698 , 0.48099533,
       0.43566814, 0.4635518 , 0.4269491 , 0.44062296, 0.42776403,
       0.46513376, 0.4213936 , 0.44270948, 0.4231782 , 0.44582006,
       0.40675786, 0.4277433 , 0.40291068, 0.41768575, 0.4008974 ,
       0.43125457, 0.39339915, 0.4099044 , 0.39313966, 0.41763872,
       0.38671115, 0.40203834, 0.3893548 , 0.4070661 , 0.38314822,
       0.40345046, 0.38255513, 0.39143324, 0.37754506, 0.3954895 ,
       0.37013033, 0.37762395, 0.3624359 , 0.37766936, 0.35681608],
dtype=np.float32).reshape(-1, 1)


def ctc_greedy_decoder_1(scores, seq_len, blank_index=0, merge_repeated=True):
    """
    Performs greedy decoding on the logits given in input (best path)

    Args:
        scores np.array[TxNxC] - input logits
        seq_len np.array[N,] - vector containing sequence lengths
        blank_index (int) - blank index number
        merge_repeated (bool) - if consecutive logits' maximum indices are the same,
            only the first of these is emitted

    Returns:
        list[np.array] - list of decoded vectors
    """
    _, N, _ = scores.shape
    output = []
    # For each batch entry, identify the transitions
    for b in range(N):
        seq_len_b = seq_len[b]
        output_b = []
        prev_class_ix = -1

        # Concatenate most probable characters per time-step which yields the best path
        # Remove duplicate characters and all blanks
        for t in range(seq_len_b):
            row = scores[t][b]
            max_class_ix = np.argmax(row)
            if (max_class_ix != blank_index and not
                    (merge_repeated and max_class_ix == prev_class_ix)):
                output_b.append(max_class_ix)
            prev_class_ix = max_class_ix
        output.append(np.array(output_b, dtype=np.int32))
    return output


def ctc_greedy_decoder(logits, seq_len, blank_label=0, merge_repeated=True):
    arg_maxes = np.argmax(logits, 2).T
    decodes = []
    for batch in arg_maxes:
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


def stack_frames(sequence, n_frames, last=False):
    """
    Stack items from a sequence into frames by `n_frames` size
    """
    stack = []
    stack_i = []
    for item in sequence:
        if len(stack_i) < n_frames:
            stack_i.append(item)
        elif len(stack_i) == n_frames:
            stack.append(np.stack(stack_i))
            stack_i = []
    if last and len(stack_i) > 0:
        stack.append(np.stack(stack_i))
    return stack


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


class Rescale:
    """
    Subtracts the frame mean and divides by the standard deviation.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, frame):
        return standardize(frame, self.mean, self.std)


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
