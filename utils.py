import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


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


def generate_chord_templates(nonchord=False, seventh=False):
    """Generate chord templates of major and minor triads (and possibly nonchord)

    Args:
        nonchord: If "True" then add nonchord template

    Returns:
        chord_templates: Matrix containing chord_templates as columns
    """
    template_cmaj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]).T
    template_cmin = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]).T
    num_chord = 24
    if nonchord:
        num_chord = 25
    chord_templates = np.ones((12, num_chord))
    for shift in range(12):
        chord_templates[:, shift] = np.roll(template_cmaj, shift)
        chord_templates[:, shift+12] = np.roll(template_cmin, shift)

    if seventh is True:
        seventh_templates = generate_seventh_chord_templates()
        chord_templates = np.hstack((chord_templates, seventh_templates))
    return chord_templates


def get_seventh_chord_labels(ext_maj7=':maj7', ext_min7=':min7', ext7=':7'):
    """Generate chord labels for major and minor seventh

    Returns:
        chord_labels: List of chord labels
    """
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_labels_maj = [s + ext_maj7 for s in chroma_labels]
    chord_labels_min = [s + ext_min7 for s in chroma_labels]
    chord_labels7 = [s + ext7 for s in chroma_labels]
    chord_labels = chord_labels_maj + chord_labels_min + chord_labels7
    return chord_labels


def generate_seventh_chord_templates():
    """Generate chord templates of major and minor seventh

    Returns:
        chord_templates: Matrix containing chord_templates as columns
    """
    # Generate maj7 template
    template_cmaj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]).T
    # Generate min7 template
    template_cmin = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]).T
    # Generate dominant seventh template
    template_dominant = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]).T
    num_chord = 36
    chord_templates = np.ones((12, num_chord))
    for shift in range(12):
        chord_templates[:, shift] = np.roll(template_cmaj, shift)
        chord_templates[:, shift+12] = np.roll(template_cmin, shift)
        chord_templates[:, shift+24] = np.roll(template_dominant, shift)
    return chord_templates


def chord_recognition_template(X, nonchord=False, seventh=False):
    """Conducts template-based chord recognition
    with major and minor triads (and possibly nonchord)

    Args:
        X: Chromagram
        nonchord: If "True" then add nonchord template

    Returns:
        chord_sim: Chord similarity matrix
        chord_max: Binarized chord similarity matrix only containing maximizing chord
    """
    chord_templates = generate_chord_templates(nonchord=nonchord, seventh=seventh)
    chord_templates_norm = normalize(chord_templates, norm='l2', axis=0)
    X_norm = normalize(X, norm='l2', axis=0)

    # Inner product of normalized vectors as a similarity measure
    chord_sim = np.matmul(chord_templates_norm.T, X_norm)
    chord_sim = normalize(chord_sim, norm='l1', axis=0)
    chord_max_index = np.argmax(chord_sim, axis=0)
    chord_max = np.zeros(chord_sim.shape).astype(np.int32)
    for n in range(chord_sim.shape[1]):
        chord_max[chord_max_index[n], n] = 1

    return chord_sim, chord_max


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
    return ann_matrix, ann_frame, ann_seg_frame, ann_seg_ind


def uniform_transition_matrix(p=0.01, N=24):
    """Computes uniform transition matrix

    Args:
        p: Self transition probability
        N: Column and row dimension

    Returns:
        A: Output transition matrix
    """
    off_diag_entries = (1 - p) / (N - 1)     # rows should sum up to 1
    A = off_diag_entries * np.ones([N, N])
    np.fill_diagonal(A, p)
    return A


def viterbi_log_likelihood(A, C, B_O):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Args:
        A: State transition probability matrix of dimension I x I
        C: Initial state distribution  of dimension I
        B_O: Likelihood matrix of dimension I x N

    Returns:
        S_opt: Optimal state sequence of length N
        S_mat: Binary matrix representation of optimal state sequence
        D_log: Accumulated log probability matrix
        E: Backtracking matrix
    """
    I = A.shape[0]  # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, 0, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    # Matrix representation of result
    S_mat = np.zeros((I, N)).astype(np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1

    return S_mat, S_opt, D_log, E
