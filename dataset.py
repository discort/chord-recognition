import itertools
import os
import os.path
from random import shuffle
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, ChainDataset, SequentialSampler

from container import ContextContainer
from utils import convert_chord_ann_matrix, get_chord_labels, read_structure_annotation,\
    convert_chord_label, convert_ann_to_seq_label

warnings.filterwarnings('ignore')


class MirexDataset(Dataset):
    def __init__(self, audio_dir, ann_dir, window_size=4096, hop_length=2048, transform=None):
        """
        Args:
            audio_dir (string): Path to audio dir
            ann_dir (string): Path to the dir with csv annotations.
        """
        self.audio_dir = audio_dir
        self.ann_dir = ann_dir
        self.window_size = window_size
        self.hop_length = hop_length
        self.ann_list = self._build_ann_list()
        self.transform = transform
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)

    def __len__(self):
        return len(os.listdir(self.ann_dir))

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.ann_list[idx] + '.mp3')
        audio_waveform, sampling_rate = librosa.core.load(audio_path, sr=None)

        # The quarter-tone spectrogram contains only bins corresponding to frequencies
        # between 65 Hz and 2100 Hz and has 24 bins per octave.
        # This results in a dimensionality of 105 bins
        chromagram = librosa.feature.chroma_stft(audio_waveform, sr=sampling_rate, norm=None,
                                                 n_fft=self.window_size, hop_length=self.hop_length,
                                                 tuning=0, n_chroma=105)
        N_X = chromagram.shape[1]
        Fs_X = sampling_rate / self.hop_length

        ann_path = os.path.join(self.ann_dir, self.ann_list[idx] + '.lab')
        ann_matrix, _, _, _ = convert_chord_ann_matrix(ann_path, self.chord_labels, Fs=Fs_X, N=N_X, last=False)

        sample = {'sample': self.ann_list[idx], 'chromagram': chromagram, 'ann_matrix': ann_matrix}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _build_ann_list(self):
        ann_list = []
        dirs = [f for f in os.listdir(self.ann_dir)]
        for file_name in dirs:
            file_path = os.path.join(self.ann_dir, file_name)
            if os.path.isfile(file_path) and file_path.endswith('.lab'):
                ann_name = file_name.replace('.lab', '')
                ann_list.append(ann_name)
        return ann_list


class MirexFameDataset(Dataset):
    def __init__(self, audio_dir, ann_dir, window_size=4096, hop_length=2048, context_size=None):
        """
        Args:
            audio_dir (string): Path to audio dir
            ann_dir (string): Path to the dir with csv annotations.
        """
        self.audio_dir = audio_dir
        self.ann_dir = ann_dir
        self.window_size = window_size
        self.hop_length = hop_length
        self.ann_list = self._build_ann_list()
        self.context_size = context_size
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)
        self._frames = []
        self._init_dataset()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    def _init_dataset(self):
        for filename in self.ann_list:
            audio_frames = self._init_audio(filename)
            self._frames.extend(audio_frames)

    def _init_audio(self, filename):
        audio_path = os.path.join(self.audio_dir, filename + '.mp3')
        audio_waveform, sampling_rate = librosa.core.load(audio_path, sr=None)

        # The quarter-tone spectrogram contains only bins corresponding to frequencies
        # between 65 Hz and 2100 Hz and has 24 bins per octave.
        # This results in a dimensionality of 105 bins
        chromagram = librosa.feature.chroma_stft(audio_waveform, sr=sampling_rate, norm=None,
                                                 n_fft=self.window_size, hop_length=self.hop_length,
                                                 tuning=0, n_chroma=105)
        N_X = chromagram.shape[1]
        Fs_X = sampling_rate / self.hop_length

        ann_path = os.path.join(self.ann_dir, filename + '.lab')
        ann_matrix, _, _, _ = convert_chord_ann_matrix(
            fn_ann=ann_path, chord_labels=self.chord_labels, Fs=Fs_X, N=N_X, last=False)

        container = ContextContainer(chromagram, self.context_size)
        result = []
        for frame, idx_target in zip(container, range(N_X)):
            #label = np.argmax(ann_matrix[:, idx_target])
            label = ann_matrix[:, idx_target].astype('long')
            if not np.any(label):  # Exclude unlabeled data (not majmin)
                continue
            result.append((frame.reshape(1, *frame.shape), label))
        return result

    def _build_ann_list(self):
        ann_list = []
        for root, dirs, files in os.walk(self.ann_dir):
            for file_name in files:
                if file_name.startswith('.'):
                    continue

                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path) and file_path.endswith('.lab'):
                    ann_name = file_name.replace('.lab', '')
                    ann_list.append(os.path.join(os.path.basename(root), ann_name))
        return ann_list


class FrameLabelDataset:
    def __init__(self, audio_dir, ann_dir, window_size=4096, hop_length=2048):
        self.audio_dir = audio_dir
        self.ann_dir = ann_dir
        self.window_size = window_size
        self.hop_length = hop_length
        self.ann_list = self._build_ann_list()
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)
        self.labels = []
        self._init_dataset()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx]

    def _init_dataset(self):
        for filename in self.ann_list:
            labels = self._get_labels(filename)
            self.labels.extend(labels)

    def _get_labels(self, filename):
        audio_path = os.path.join(self.audio_dir, filename + '.mp3')
        _, sampling_rate = librosa.core.load(audio_path, sr=None)
        Fs_X = sampling_rate / self.hop_length

        ann_path = os.path.join(self.ann_dir, filename + '.lab')
        ann_seg_ind = read_structure_annotation(ann_path, Fs=Fs_X, index=True)
        ann_seg_ind = convert_chord_label(ann_seg_ind)
        result = convert_ann_to_seq_label(ann_seg_ind)
        return result

    def _build_ann_list(self):
        ann_list = []
        for root, dirs, files in os.walk(self.ann_dir):
            for file_name in files:
                if file_name.startswith('.'):
                    continue

                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path) and file_path.endswith('.lab'):
                    ann_name = file_name.replace('.lab', '')
                    ann_list.append(os.path.join(os.path.basename(root), ann_name))
        return ann_list


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


if __name__ == '__main__':
    # Some testing stuff
    from torch.utils.data import DataLoader, ConcatDataset
    from augmentations import SemitoneShift

    dataset = MirexFameDataset(audio_dir='data/beatles/mp3s-32k/Let_It_Be/',
                               ann_dir='data/beatles/chordlabs/Let_It_Be/',
                               window_size=8192, hop_length=4096, context_size=7)
    import pudb; pudb.set_trace()
    augmented = SemitoneShift(dataset, p=1.0, max_shift=4, bins_per_semitone=2)
    dataset = ConcatDataset([dataset, augmented])
    loader_train = DataLoader(dataset, shuffle=True, num_workers=0, batch_size=32)
    for inputs, labels in loader_train:
        print(inputs.shape, labels.shape)
