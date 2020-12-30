import csv
import itertools
import os
import os.path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler

from .augmentations import SemitoneShift, DetuningShift, one_hot, shift_majmin_targets
from .cache import HDF5Cache
from .dataset import ChromaDataset, context_window
from .dataset import prepare_datasource, collect_files
from .ann_utils import get_chord_labels, convert_annotation_matrix,\
    convert_label_sequence, read_structure_annotation, convert_chord_label


def save_data(df, dir_path='data/augmented'):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = os.path.join(dir_path, 'augmented_semitone.csv')

    df.to_csv(
        path_or_buf=filename,
        columns=('label',),
        header=False,
        index=False)
    print(f"Created '{filename}'")


def _mark_to_remove(row):
    if row.name == 0:  # Skip first row
        return 0
    if row.label == row.prev_label:
        return 1
    return 0


def make_frame_df(data_source):
    sampler = SequentialSampler(data_source)
    chord_labels = get_chord_labels(nonchord=True)
    frame_labels = []
    for idx in sampler:
        data, labels = sampler.data_source[idx]
        if labels.ndim == 2:
            for i in range(labels.shape[0]):
                if not labels[i, :].any():
                    continue
                frame_label = next(itertools.compress(chord_labels, labels[i, :]))
                frame_labels.append(frame_label)
        elif labels.ndim == 1:
            frame_label = next(itertools.compress(chord_labels, labels))
            frame_labels.append(frame_label)
        else:
            raise ValueError("Dimensions must be less than 2")

    df = pd.DataFrame(frame_labels, columns=('label',))
    df['prev_label'] = df.label.shift(1)
    # df['to_remove'] = df.apply(lambda x: _mark_to_remove(x), axis=1)
    # df = df[df.to_remove == 0]
    return df


def print_audio_sample_rate(datasource=None):
    """
    Print an audio name and it's sampling_rate.
    Make sure that beatles audios have 16000 Hz but all others have 44100Hz.
    """
    from chord_recognition.dataset import prepare_datasource
    from chord_recognition.utils import read_audio
    if datasource is None:
        datasource = [audio_path for _, audio_path in prepare_datasource(
            ('queen', 'beatles', 'robie_williams', 'zweieck'))]
    for audio_path in datasource:
        audio_name = audio_path.replace('/Users/discort/ml-course/chord-recognition/data/', '')
        _, sampling_rate = read_audio(audio_path, sr=None, mono=True)
        print(f"Fs for {audio_name}: {sampling_rate}")
        assert sampling_rate == 44100


def pitch_shift(annotation_path, audio_path, shift, output_dir='data/augmented'):
    """
    Given an audio/annotation files.
    Pitch shift the audio/annotation accordingly and save them to files.
    """
    # '/Users/discort/ml-course/chord-recognition/data/beatles/chordlabs/Help_/01-Help_.lab',
    # '/Users/discort/ml-course/chord-recognition/data/beatles/mp3/Help_/01-Help_.mp3'
    audio_dir = os.path.join(output_dir, 'mp3')
    ann_dir = os.path.join(output_dir, 'chordlabs')
    if not os.path.exists(output_dir):
        os.makedirs(audio_dir)
        os.makedirs(ann_dir)
    audio_filename, ann_filename = os.path.basename(audio_path), os.path.basename(annotation_path)
    pitch_shift_audio(audio_path, shift, os.path.join(audio_dir, audio_filename))
    pitch_shift_annotation(annotation_path, shift, os.path.join(ann_dir, ann_filename))


def one_hot_ann_matrix(ann_path):
    chord_labels = get_chord_labels(nonchord=True)
    ann_seg_sec = read_structure_annotation(ann_path)
    ann_seg_sec = convert_chord_label(ann_seg_sec)
    N = len(ann_seg_sec)
    num_chords = len(chord_labels)
    ann_matrix = np.zeros((num_chords, N), dtype=np.int32)
    for n in range(N):
        label = ann_seg_sec[n][-1]
        # Generates a one-entry only for labels that are contained in "chord_labels"
        if label in chord_labels:
            label_index = chord_labels.index(label)
            ann_matrix[label_index, n] = 1
    return ann_matrix, ann_seg_sec


def pitch_shift_annotation(ann_path, shift, output_filename):
    """
    - make one-hot ann_matrix (num_chords, N)
    - make pitch-shift
    - one-hot matrix to chords sequence (use no-chord class for all-zeros as well)
    - replace sequence of initial chords by new generated sequence.
        Do not replace values if corresponded label in new seq is no-chord class
    - save a new annotation to .lab file
    """
    ann_matrix, ann_seg_sec = one_hot_ann_matrix(ann_path)
    new_ann_matrix = np.zeros_like(ann_matrix)
    shifts = np.array([shift])
    for i in range(ann_matrix.shape[1]):
        # Do not shift if there is no label
        if not ann_matrix[:, i].any():
            continue
        new_ann_matrix[:, i] = shift_majmin_targets(ann_matrix[:, i], shifts)

    label_seq = convert_annotation_matrix(new_ann_matrix, nonchord=True)
    assert len(ann_seg_sec) == len(label_seq)
    for i in range(len(ann_seg_sec)):
        if label_seq[i] != 'N':
            ann_seg_sec[i][-1] = label_seq[i]

    with open(output_filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for ann in ann_seg_sec:
            writer.writerow(ann)


def pitch_shift_audio(audio_path, shift, output_filename):
    """
    # ffmpeg -i <filename> -filter_complex
        "asetrate=<sample_rate>*2^(<pitch>/12),atempo=1/2^(<pitch>/12)" "output.mp3"
    """
    import subprocess
    sampling_rate = 41000
    # subprocess.call(
    #     ['ffmpeg', '-i', audio_path, '-filter_complex',
    #      f'asetrate={sampling_rate}*2^({shift}/12),atempo=1/2^({shift}/12)', output_filename]
    # )
    # Sox accepts accepts cents. 100ths of a semitone.
    # There are 12 semitones to an octave, so that would mean ±1200 as a parameter.
    shift = shift * 100
    subprocess.call(
        ['sox', audio_path, output_filename, 'pitch', str(shift)]
    )


def pitch_shift_data(datasource, max_shift=4):
    shifts = np.random.randint(-max_shift, max_shift + 1, len(datasource))
    for shift, (ann_path, audio_path) in zip(shifts, datasource):
        pitch_shift(ann_path, audio_path, shift)


def main():
    from sklearn.model_selection import train_test_split
    from imblearn.datasets import make_imbalance
    ds = prepare_datasource(('beatles', ))

    dataset = ChromaDataset(
        ds, window_size=8192, hop_length=4096,
        context_size=0,
        cache=HDF5Cache('chroma_cache.hdf5'))
    # sampling_strategy = {
    #     0: 10000,
    #     2: 10000,
    #     4: 10000,
    #     7: 10000,
    #     9: 10000,
    #     11: 10000,
    #     24: 10000,
    # }
    # # reshape X to have (n_samples, n_features)
    # X = np.vstack([x.reshape(1, -1) for x, _ in dataset])
    # y = [yi for _, yi in dataset]
    # X, y = make_imbalance(X, y,
    #                       sampling_strategy=sampling_strategy)
    # # Reshape X to it's original form
    # X = [x.reshape(1, 105, 15) for x, _ in X.shape[0]]
    # train_dataset[43]
    # loader_train = DataLoader(dataset, shuffle=True, num_workers=0, batch_size=32)

    # df = make_frame_df(augmented)
    # save_data(df)


if __name__ == '__main__':
    main()
