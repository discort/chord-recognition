import itertools

import numpy as np

from .utils import read_csv


DEFAULT_EXT_MINOR = 'm'


def get_chord_labels(ext_minor=None, nonchord=False):
    """Generate chord labels for major and minor triads (and possibly non-chord label)

    Args:
        nonchord: If "True" then add nonchord template

    Returns:
        chord_labels: List of chord labels
    """
    if ext_minor is None:
        ext_minor = DEFAULT_EXT_MINOR
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_labels_maj = chroma_labels
    chord_labels_min = [s + ext_minor for s in chroma_labels]
    chord_labels = chord_labels_maj + chord_labels_min
    if nonchord is True:
        chord_labels += ['N']
    return chord_labels


def build_ann_df(fn_ann, sep=None):
    df = read_csv(fn_ann, sep=sep)
    if len(df.columns) < 3:
        # Some annotations are separated by space but others by \t
        df = read_csv(fn_ann, sep='\t')
    df.columns = ['start', 'end', 'label']
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

    df = build_ann_df(fn_ann)
    df = convert_chord_label(df)
    ann = [(start, end, label) for i, (start, end, label) in df.iterrows()]
    ann = convert_structure_annotation(ann, Fs=Fs, remove_digits=remove_digits, index=index)
    return ann


def convert_chord_label(df):
    """Replace for segment-based annotation in each chord label the string ':min' by 'm'
    and convert flat chords into sharp chords using enharmonic equivalence

    Check MIREX chord vocabularies for details:
    https://www.music-ir.org/mirex/wiki/2020:Audio_Chord_Estimation#Chord_Vocabularies
    """
    df = df.copy()
    df['label'] = df.label.str.replace('Cb', 'B', regex=False)
    df['label'] = df.label.str.replace('Db', 'C#', regex=False)
    df['label'] = df.label.str.replace('Eb', 'D#', regex=False)
    df['label'] = df.label.str.replace('Fb', 'E', regex=False)
    df['label'] = df.label.str.replace('Gb', 'F#', regex=False)
    df['label'] = df.label.str.replace('Ab', 'G#', regex=False)
    df['label'] = df.label.str.replace('Bb', 'A#', regex=False)

    df['label'] = df.label.str.replace(r'^C:maj$', 'C', regex=True)
    df['label'] = df.label.str.replace(r'^C#:maj$', 'C#', regex=True)
    df['label'] = df.label.str.replace(r'^D:maj$', 'D', regex=True)
    df['label'] = df.label.str.replace(r'^D#:maj$', 'D#', regex=True)
    df['label'] = df.label.str.replace(r'^E:maj$', 'E', regex=True)
    df['label'] = df.label.str.replace(r'^F:maj$', 'F', regex=True)
    df['label'] = df.label.str.replace(r'^F#:maj$', 'F#', regex=True)
    df['label'] = df.label.str.replace(r'^G:maj$', 'G', regex=True)
    df['label'] = df.label.str.replace(r'^G#:maj$', 'G#', regex=True)
    df['label'] = df.label.str.replace(r'^A:maj$', 'A', regex=True)
    df['label'] = df.label.str.replace(r'^A#:maj$', 'A#', regex=True)
    df['label'] = df.label.str.replace(r'^B:maj$', 'B', regex=True)

    df['label'] = df.label.str.replace(':maj7', 'maj7', regex=False)
    df['label'] = df.label.str.replace(':min7', 'min7', regex=False)
    df['label'] = df.label.str.replace(':min', 'm', regex=True)
    df['label'] = df.label.str.replace(':7', '7', regex=True)
    return df


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
    ann_seg_ind = read_structure_annotation(fn_ann, Fs=Fs, index=True)

    ann_frame = convert_ann_to_seq_label(ann_seg_ind)
    if N is None:
        N = len(ann_frame)
    if N < len(ann_frame):
        ann_frame = ann_frame[:N]
    if N > len(ann_frame):
        if last is True:
            pad_symbol = ann_frame[-1]
        else:
            pad_symbol = 'N'
        ann_frame = ann_frame + [pad_symbol] * (N - len(ann_frame))
    ann_seg_frame = convert_sequence_ann(ann_frame, Fs=1)

    num_chords = len(chord_labels) + 1
    ann_matrix = np.zeros((num_chords, N), dtype=np.int64)
    for n in range(N):
        label = ann_frame[n]
        # Generates a one-hot representation only for labels that are contained in "chord_labels"
        if label in chord_labels:
            label_index = chord_labels.index(label) + 1
            ann_matrix[label_index, n] = 1
    return ann_matrix, ann_frame, ann_seg_frame, ann_seg_ind, ann_seg_sec


def convert_annotation_matrix(ann_matrix, ext_minor=None, nonchord=True):
    """Converts annotation matrix to sequence of labels

    Args:
        ann_matrix: annotation matrix (N x num_classes)

    Returns:
        labels_sec: list of labels sequences
    """
    chord_labels = get_chord_labels(ext_minor, nonchord=nonchord)

    labels_sec = []
    N = ann_matrix.shape[0]
    for i in range(N):
        one_hot = ann_matrix[i, :]
        label = list(itertools.compress(chord_labels, one_hot))
        if not len(label):
            raise ValueError("Invalid chord decoding")
        label = label[0]
        labels_sec.append(label)
    return labels_sec


def convert_onehot_ann(ann_matrix, hop_length, sr, ext_minor=None, nonchord=False):
    """Convert annotation matrix to basic ann representation

    # Convert one-hot repr to label repr (25, 2822) -> (2822, 1)
    # Convert sequence annotation list ([s,t,'label'])
    # Convert list to structure annotation [3055, 3076, 'N'] -> (283.724286, 285.666644, 'N')]

    Args:
        ann_matrix: N x num_classes
        hop_length: number audio of frames between STFT columns
    """
    label_seq = convert_annotation_matrix(ann_matrix, ext_minor=ext_minor, nonchord=True)
    ann_seg = convert_label_sequence(label_seq)
    Fs_X = sr / hop_length
    ann = convert_annotation_segments(ann_seg, Fs=Fs_X)
    return ann


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


def convert_annotation_segments(ann_seg, Fs=1, round_decimals=4):
    """Converts annotation segments to time-wised notation

    Args:end_index
        ann_seg: segment-based annotation in [s,t,'label'] format

    Returns:
        result: time-based annotation in [start_time,end_time,'label'] format
    """
    result = []
    for r in ann_seg:
        s = r[0] / Fs
        t = r[1] / Fs
        if round_decimals:
            s = np.around(s, decimals=round_decimals)
            t = np.around(t, decimals=round_decimals)
        result.append((s, t, r[2]))
    return result
