import itertools
import os
import os.path

import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler

from augmentations import SemitoneShift
from dataset import MirexFameDataset, make_frame_df, FrameLabelDataset
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
