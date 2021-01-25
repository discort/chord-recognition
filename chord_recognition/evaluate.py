import argparse
import csv
import itertools
import os

import mir_eval
import numpy as np
import matplotlib.pyplot as plt

from chord_recognition.cache import HDF5Cache
from chord_recognition.dataset import prepare_datasource, SpecDataset, collect_files
from chord_recognition.predict import ChordRecognition, DeepChordRecognition


def compute_cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate character error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.

    Args:
        reference (basestring): The reference sentence.
        hypothesis (basestring): The hypothesis sentence.
        ignore_case (bool): Whether case-sensitive or not.
        remove_space (bool): Whether remove internal space characters

    Returns:
        (float): Character error rate.

    raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.

    Args:
        reference (basestring): The reference sentence.
        hypothesis (basestring): The hypothesis sentence.
        ignore_case (bool): Whether case-sensitive or not.
        remove_space (bool): Whether remove internal space characters

    Returns:
        (list): Levenshtein distance and length of reference sentence.
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def compute_wer(reference, hypothesis, ignore_case=False):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER.

    Args:
        reference (list[str]): The reference sentence.
        hypothesis (list[str]): The hypothesis sentence.
        ignore_case (bool): Whether case-sensitive or not.
    Returns:
        float - Word error rate.

    raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def word_errors(reference, hypothesis, ignore_case=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    Args:
        reference (list[str]): The reference sentence.
        hypothesis (list[str]): The hypothesis sentence.
        ignore_case (bool): Whether case-sensitive or not.
    Returns:
        (list) - Levenshtein distance and word number of reference sentence.
    """
    if ignore_case is True:
        reference = [ref.lower() for ref in reference]
        hypothesis = [hyp.lower() for hyp in hypothesis]

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def plot_confusion_matrix(
        cm, classes, normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues,
        fontsize=10):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_eval_measures(I_ref, I_est):
    """Compute evaluation measures including precision, recall, and F-measure

    Args:
        I_ref, I_est: Sets of positive items for reference and estimation

    Returns:
        P, R, F: Precsion, recall, and F-measure
        TP, FP, FN: Number of true positives, false positives, and false negatives
    """
    assert I_ref.shape == I_est.shape, "Dimension of input matrices must agree"
    TP = np.sum(np.logical_and(I_ref, I_est))
    FP = np.sum(I_est > 0, axis=None) - TP
    FN = np.sum(I_ref > 0, axis=None) - TP
    P = 0
    R = 0
    F = 0
    if TP > 0:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F = 2 * P * R / (P + R)
    return P, R, F, TP, FP, FN


def compute_scores(annotation_files, prediction_files):
    assert len(annotation_files) == len(prediction_files)
    assert len(annotation_files) > 0

    scores = []
    total_length = 0.

    for af, pf in zip(annotation_files, prediction_files):
        ann_int, ann_lab = mir_eval.io.load_labeled_intervals(af)
        pred_int, pred_lab = mir_eval.io.load_labeled_intervals(pf)

        # we assume that the end-time of the last annotated label is the
        # length of the song
        song_length = ann_int[-1][1]
        total_length += song_length

        scores.append(
            (pf, song_length,
             mir_eval.chord.evaluate(ann_int, ann_lab, pred_int, pred_lab))
        )

    return scores, total_length


def print_scores(scores):
    for name, val in scores.items():
        label = '\t{}:'.format(name).ljust(16)
        print(label + '{:.3f}'.format(val))


def average_scores(scores, total_length):
    # initialise the average score with all metrics and values 0.
    avg_score = {metric: 0. for metric in scores[0][-1]}

    for _, length, score in scores:
        weight = length / total_length
        for metric in score:
            avg_score[metric] += float(weight * score[metric])

    return avg_score


def compute_average_scores(annotation_files, prediction_files):
    # first, compute all individual scores
    scores, total_length = compute_scores(annotation_files, prediction_files)
    return average_scores(scores, total_length)


def save_annotations(annotations, file_path):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(file_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for ann in annotations:
            writer.writerow(ann)


def evaluate_dataset(dataset, save_ann=False, result_dir='results'):
    estimator = DeepChordRecognition(
        window_size=dataset.window_size,
        hop_length=dataset.hop_length,
        ext_minor=':min',
        postprocessing=None,
        nonchord=True)

    for i in range(len(dataset)):
        ann_path = dataset.datasource[i][0]
        sample_name = ann_path.split('/')[-1].replace('.lab', '')
        spec, ann_matrix = dataset[i]

        spec = estimator.preprocess(spec)
        out = estimator.predict_labels(spec)
        out = estimator.postprocess(out)

        print(f'Eval: <{sample_name}>')
        if save_ann:
            result_path = '/'.join(ann_path.split('/')[-4:]).replace('/chordlabs', '')
            result_path = os.path.join(result_dir, result_path)
            result_ann = estimator.decode_chords(out, 44100)
            save_annotations(result_ann, result_path)


def print_ds_compute_average_scores(ds_name):
    excluded_files = (
        # robbie
        "10-She's The One",
        "09-Knutsford City Limits",
        # beatles
        "03-You_Won_t_See_Me",
        "04-Nowhere_Man",
        "02-Dear_Prudence",
        # zweieck
        "16_-_Zu_Leise_Für_Mich.lab",
    )
    ann_root = f'data/{ds_name}/chordlabs/'
    annotation_files = collect_files(ann_root, excluded_files=excluded_files)
    pred_root = f'results/{ds_name}/'
    prediction_files = collect_files(pred_root, ext='', excluded_files=excluded_files)
    scores = compute_average_scores(annotation_files, prediction_files)
    print_scores(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds", help="Specify the datasource names for evaluations",
        nargs='?', action='append')
    args = parser.parse_args()
    ds_names = args.ds

    for ds_name in ds_names:
        print(f"Evaluating {ds_name} ...")
        datasource = prepare_datasource((ds_name,))
        dataset = SpecDataset(
            datasource=datasource,
            window_size=8192,
            hop_length=4410,
            cache=HDF5Cache('spectrogram_ann_cache.hdf5'))
        evaluate_dataset(
            dataset=dataset,
            save_ann=True)
        print_ds_compute_average_scores(ds_name)


if __name__ == '__main__':
    main()
