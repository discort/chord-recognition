import csv
import itertools
import os

import mir_eval
import numpy as np
import matplotlib.pyplot as plt

from chord_recognition.ann_utils import convert_onehot_ann
from chord_recognition.cache import HDF5Cache
from chord_recognition.models import deep_auditory_v2
from chord_recognition.dataset import prepare_datasource, ChromaDataset, collect_files
from chord_recognition.predict import predict_annotations


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


def evaluate_dataset(dataset, model, save_ann=False, result_dir='results'):
    total_R = total_P = total_F = 0.0
    total_count = 0
    for i in range(len(dataset)):
        ann_path = dataset.datasource[i][0]
        sample_name = ann_path.split('/')[-1].replace('.lab', '')
        spec, ann_matrix = dataset[i]
        out = predict_annotations(spec, model)
        P, R, F1, TP, FP, FN = compute_eval_measures(ann_matrix, out.T)
        title = (f'Eval: <{sample_name}> N={out.shape[0]} TP={TP} FP={FP} FN={FN}'
                 f' P={P:.3f} R={R:.3f} F1={F1:.3f}')
        print(title)

        if save_ann:
            result_path = '/'.join(ann_path.split('/')[-4:]).replace('/chordlabs', '')
            result_path = os.path.join(result_dir, result_path)
            result_ann = convert_onehot_ann(
                out, dataset.hop_length, 44100, ext_minor=':min', nonchord=True)
            save_annotations(result_ann, result_path)

        total_count += 1
        total_R += R
        total_P += P
        total_F += F1

    total_R /= total_count
    total_P /= total_count
    total_F /= total_count
    print(f'Total R: {total_R}')
    print(f'Total P: {total_P}')
    print(f'Total F: {total_F}')


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
    model = deep_auditory_v2(pretrained=True, model_name='deep_auditory_v2_exp4_3.pth')
    model.eval()  # set model to evaluation mode

    datasource = prepare_datasource(('robbie_williams',))
    dataset = ChromaDataset(
        datasource=datasource,
        window_size=8192,
        hop_length=4096,
        context_size=None,
        cache=HDF5Cache('chroma_cache.hdf5'))
    evaluate_dataset(
        dataset=dataset,
        model=model,
        save_ann=True)
    ds_name = 'robbie_williams'
    print_ds_compute_average_scores(ds_name)


if __name__ == '__main__':
    main()
