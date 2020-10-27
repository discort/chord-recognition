import itertools
import os
import os.path

import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler

from augmentations import SemitoneShift
from dataset import MirexFameDataset
from utils import get_chord_labels


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
        frame_label = next(itertools.compress(chord_labels, labels))
        frame_labels.append(frame_label)

    df = pd.DataFrame(frame_labels, columns=('label',))
    df['prev_label'] = df.label.shift(1)
    df['to_remove'] = df.apply(lambda x: _mark_to_remove(x), axis=1)
    # df = df[df.to_remove == 0]
    return df


def main():
    dataset = MirexFameDataset(audio_dir='data/beatles/mp3s-32k/',
                               ann_dir='data/beatles/chordlabs/',
                               window_size=8192, hop_length=4096, context_size=7)
    queen = MirexFameDataset(audio_dir='data/queen/mp3/',
                             ann_dir='data/queen/chordlabs/',
                             window_size=8192, hop_length=4096, context_size=7)
    dataset = ConcatDataset([dataset, queen])
    augmented = SemitoneShift(dataset, p=1.0, max_shift=4, bins_per_semitone=2)
    df = make_frame_df(augmented)
    save_data(df)


if __name__ == '__main__':
    main()
