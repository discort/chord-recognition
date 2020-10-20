import itertools
import tempfile

import librosa
import numpy as np
import pandas as pd


def get_chord_labels(ext_minor='m', nonchord=False):
    """Generate chord labels for major and minor triads (and possibly non-chord label)

    Args:
        nonchord: If "True" then add nonchord template

    Returns:
        chord_labels: List of chord labels
    """
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_labels_maj = chroma_labels
    chord_labels_min = [s + ext_minor for s in chroma_labels]
    chord_labels = chord_labels_maj + chord_labels_min
    if nonchord is True:
        chord_labels += ['N']
    return chord_labels


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


def convert_ann_to_seq_label(ann):
    """Convert structure annotation with integer time positions (given in indices)
    into label sequence

    Args:
        ann: Annotation (list  [[s,t,'label'], ...], with s,t being integers)

    Returns:
        X: Sequencs of labels
    """
    X = []
    for seg in ann:
        K = seg[1] - seg[0]
        for k in range(K):
            X.append(seg[2])
    return X


def convert_structure_annotation(ann, Fs=1, remove_digits=False, index=False):
    """Convert structure annotations

    Args:
        ann: Structure annotions
        Fs: Sampling rate
        remove_digits: Remove digits from labels

    Returns:
        ann_converted: Converted annotation
    """
    ann_converted = []
    for r in ann:
        s = r[0] * Fs
        t = r[1] * Fs
        if index:
            s = int(np.round(s))
            t = int(np.round(t))
        if remove_digits:
            label = ''.join([i for i in r[2] if not i.isdigit()])
        else:
            label = r[2]
        ann_converted = ann_converted + [[s, t, label]]
    return ann_converted


def read_structure_annotation(fn_ann, Fs=1, remove_digits=False, index=False):
    """Read and convert structure annotation,

    Args:
        fn_ann: path and filename for structure annotions
        Fs: Sampling rate
        remove_digits: Remove digits from labels

    Returns:
        ann: Annotations
    """

    df = read_csv(fn_ann)
    if len(df.columns) < 3:
        # Some annotations are separated by space but others by \t
        df = read_csv(fn_ann, sep='\t')
    ann = [(start, end, label) for i, (start, end, label) in df.iterrows()]
    ann = convert_structure_annotation(ann, Fs=Fs, remove_digits=remove_digits, index=index)
    return ann


def convert_chord_label(ann):
    """Replace for segment-based annotation in each chord label the string ':min' by 'm'
    and convert flat chords into sharp chords using enharmonic equivalence

    Args:
        ann: Segment-based annotation with chord labels

    Returns:
        ann: Segment-based annotation with chord labels
    """
    for k in range(len(ann)):
        ann[k][2] = ann[k][2].replace(':min', 'm')
        ann[k][2] = ann[k][2].replace('Db', 'C#')
        ann[k][2] = ann[k][2].replace('Eb', 'D#')
        ann[k][2] = ann[k][2].replace('Gb', 'F#')
        ann[k][2] = ann[k][2].replace('Ab', 'G#')
        ann[k][2] = ann[k][2].replace('Bb', 'A#')

        ann[k][2] = ann[k][2].replace('C:maj', 'C')
        ann[k][2] = ann[k][2].replace('C#:maj', 'C#')
        ann[k][2] = ann[k][2].replace('D:maj', 'D')
        ann[k][2] = ann[k][2].replace('D#:maj', 'D#')
        ann[k][2] = ann[k][2].replace('E:maj', 'E')
        ann[k][2] = ann[k][2].replace('F:maj', 'F')
        ann[k][2] = ann[k][2].replace('F#:maj', 'F#')
        ann[k][2] = ann[k][2].replace('G:maj', 'G')
        ann[k][2] = ann[k][2].replace('G#:maj', 'G#')
        ann[k][2] = ann[k][2].replace('A:maj', 'A')
        ann[k][2] = ann[k][2].replace('A#:maj', 'A#')
        ann[k][2] = ann[k][2].replace('B:maj', 'B')
    return ann


def convert_sequence_ann(seq, Fs=1):
    """Convert label sequence into segment-based annotation

    Args:
        seq: Label sequence
        Fs: Feature rate

    Returns:
        ann: Segment-based annotation for label sequence
    """
    ann = []
    for m in range(len(seq)):
        ann.append([(m - 0.5) / Fs, (m + 0.5) / Fs, seq[m]])
    return ann


def convert_chord_ann_matrix(fn_ann, chord_labels, Fs=1, N=None, last=False):
    """Convert segment-based chord annotation into various formats

    Args:
        fn_ann: Filename of segment-based chord annotation
        chord_labels: List of chord labels
        Fs: Feature rate
        N: Number of frames to be generated (by cutting or extending)
           Only enforced for ann_matrix, ann_frame, ann_seg_frame
        last: If 'True' uses for extension last chord label, otherwise uses nonchord label 'N'

    Returns:
        ann_matrix: Encoding of label sequence in form of a binary time-chord representation
        ann_frame: Label sequence (specified on the frame level)
        ann_seg_frame: Encoding of label sequence as segment-based annotation (given in indices)
        ann_seg_ind: Segment-based annotation with segments (given in indices)
    """
    ann_seg_sec = read_structure_annotation(fn_ann)
    ann_seg_sec = convert_chord_label(ann_seg_sec)
    ann_seg_ind = read_structure_annotation(fn_ann, Fs=Fs, index=True)
    ann_seg_ind = convert_chord_label(ann_seg_ind)

    ann_frame = convert_ann_to_seq_label(ann_seg_ind)
    if N is None:
        N = len(ann_frame)
    if N < len(ann_frame):
        ann_frame = ann_frame[:N]
    if N > len(ann_frame):
        if last == True:
            pad_symbol = ann_frame[-1]
        else:
            pad_symbol = 'N'
        ann_frame = ann_frame + [pad_symbol] * (N - len(ann_frame))
    ann_seg_frame = convert_sequence_ann(ann_frame, Fs=1)

    num_chords = len(chord_labels)
    ann_matrix = np.zeros((num_chords, N))
    for n in range(N):
        label = ann_frame[n]
        # Generates a one-entry only for labels that are contained in "chord_labels"
        if label in chord_labels:
            label_index = chord_labels.index(label)
            ann_matrix[label_index, n] = 1
    return ann_matrix, ann_frame, ann_seg_frame, ann_seg_ind, ann_seg_sec


def compute_chromagram(audio_waveform, Fs, window_size=8192, hop_length=4096,
                       n_chroma=105, norm=None, tuning=0):
    """
    Computes a chromagram from a waveform

    The quarter-tone spectrogram contains only bins corresponding to frequencies
    between 65 Hz and 2100 Hz and has 24 bins per octave.
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


def read_audio(path, Fs=None, mono=False):
    """Reads an audio file

    Args:
        path: Path to audio file
        Fs: Resample audio to given sampling rate. Use native sampling rate if None.
        mono (bool): Convert multi-channel file to mono.

    Returns:
        x: Audio time series (np.ndarray [shape=(n,))
        Fs: Sampling rate
    """
    return librosa.load(path, sr=Fs, mono=mono)


def read_audio_from_stream(stream, Fs=None, mono=False):
    """Reads an audio file from stream.
    Since librosa does not support reading mp3 in buffer.
    https://github.com/librosa/librosa/pull/1066

    Args:
        path: Path to audio file
        Fs: Resample audio to given sampling rate. Use native sampling rate if None.
        mono: Convert multi-channel file to mono (bool)

    Returns:
        x: Audio time series (np.ndarray [shape=(n,))
        Fs: Sampling rate
    """
    with tempfile.NamedTemporaryFile() as ntf:
        ntf.write(stream.read())
        return read_audio(ntf.name, Fs, mono)


def compute_annotation(ann_matrix, hop_length, Fs, nonchord=False):
    # Convert one-hot repr to label repr (25, 2822) -> (2822, 1)
    # Convert sequence annotation list ([s,t,'label'])
    # Convert list to structure annotation [3055, 3076, 'N'] -> (283.724286, 285.666644, 'N')]
    label_seq = convert_annotation_matrix(ann_matrix, nonchord=True)
    ann_seg = convert_label_sequence(label_seq)
    Fs_X = Fs / hop_length
    ann = convert_annotation_segments(ann_seg, Fs=Fs_X)
    return ann


def convert_annotation_matrix(ann_matrix, nonchord=False):
    """Converts annotation matrix to sequence of labels

    Args:
        ann_matrix: annotation matrix

    Returns:
        labels_sec: list of labels sequences
    """
    chord_labels = get_chord_labels(nonchord=nonchord)

    labels_sec = []
    N = ann_matrix.shape[1]
    for i in range(N):
        one_hot = ann_matrix[:, i]
        label = next(itertools.compress(chord_labels, one_hot))
        labels_sec.append(label)
    return labels_sec


def convert_label_sequence(label_seq):
    """Converts label sequence to frame-segment representation [s,t,'label']

    Args:
        label_seq: list of labels

    Returns:
        ann_seg: segment-based annotation
    """
    result = []
    N = len(label_seq)
    start_index = end_index = 0
    for i in range(N):
        if i == 0:
            end_index += 1
            continue

        item = label_seq[i]
        prev_item = label_seq[i - 1]

        if item != prev_item:
            result.append((start_index, end_index, prev_item))
            start_index = i
            end_index += 1
        else:
            end_index += 1
            if i == N - 1:
                result.append((start_index, end_index, item))
    return result


def convert_annotation_segments(ann_seg, Fs=1):
    """Converts annotation segments to time-wised notation

    Args:
        ann_seg: segment-based annotation in [s,t,'label'] format

    Returns:
        result: time-based annotation in [start_time,end_time,'label'] format
    """
    result = []
    for r in ann_seg:
        s = r[0] / Fs
        t = r[1] / Fs
        result.append((s, t, r[2]))
    return result
