import itertools
import os
import os.path
from random import shuffle
import warnings

import librosa
import torch
from torch.utils.data import Dataset, IterableDataset, ChainDataset

from container import ContextContainer
from utils import convert_chord_ann_matrix, get_chord_labels

warnings.filterwarnings('ignore')


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return {'sample': sample['sample'],
                'chromagram': torch.from_numpy(sample['chromagram']),
                'ann_matrix': torch.from_numpy(sample['ann_matrix'])}


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


class FrameIterableDataset(IterableDataset):
    def __init__(self, audio_dir, ann_dir, window_size=4096, hop_length=2048, transform=None):
        """
        Args:
            audio_dir (string): Path to audio dir
            ann_dir (string): Path to the dir with csv annotations.
        """
        super(FrameIterableDataset).__init__()
        self.audio_dir = audio_dir
        self.ann_dir = ann_dir
        self.window_size = window_size
        self.hop_length = hop_length
        self.ann_list = self._build_ann_list()
        self.transform = transform
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)

    def __len__(self):
        return len(self.ann_list)

    def __iter__(self):
        self._init_iterator()
        signal_processors = [self._process_signal(ann_name) for ann_name in self._iterator]
        return itertools.chain(*signal_processors)

    def _process_signal(self, ann_name):
        audio_path = os.path.join(self.audio_dir, ann_name + '.mp3')
        audio_waveform, sampling_rate = librosa.core.load(audio_path, sr=None)

        # The quarter-tone spectrogram contains only bins corresponding to frequencies
        # between 65 Hz and 2100 Hz and has 24 bins per octave.
        # This results in a dimensionality of 105 bins
        chromagram = librosa.feature.chroma_stft(audio_waveform, sr=sampling_rate, norm=None,
                                                 n_fft=self.window_size, hop_length=self.hop_length,
                                                 tuning=0, n_chroma=105)
        N_X = chromagram.shape[1]
        Fs_X = sampling_rate / self.hop_length

        ann_path = os.path.join(self.ann_dir, ann_name + '.lab')
        ann_matrix, _, _, _ = convert_chord_ann_matrix(ann_path, self.chord_labels, Fs=Fs_X, N=N_X, last=False)

        container = ContextContainer(chromagram, 7)

        for frame, idx_target in zip(container, range(N_X)):
            yield frame.reshape(1, *frame.shape), ann_matrix[:, idx_target]

    def _init_iterator(self):
        shuffle(self.ann_list)
        self._iterator = iter(self.ann_list)

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


class MirexChainDataset(ChainDataset):
    def __init__(self, datasets):
        super(MirexChainDataset, self).__init__(datasets)
        self.ann_list = []
        for d in datasets:
            self.ann_list.extend(d.ann_list)


if __name__ == '__main__':
    # Some testing stuff
    from torch.utils.data import DataLoader

    dataset = FrameIterableDataset(audio_dir='data/beatles/mp3s-32k/Let_It_Be/',
                                   ann_dir='data/beatles/chordlabs/Let_It_Be/',
                                   window_size=8192, hop_length=4096)
    beatles_dataset = FrameIterableDataset(audio_dir='data/beatles/mp3s-32k/',
                                           ann_dir='data/beatles/chordlabs/',
                                           window_size=8192, hop_length=4096)
    queen_dataset = FrameIterableDataset(audio_dir='data/queen/mp3/',
                                         ann_dir='data/queen/chordlabs/',
                                         window_size=8192, hop_length=4096)
    dataset = MirexChainDataset([beatles_dataset, queen_dataset])
    loader_train = DataLoader(dataset, num_workers=0, batch_size=32)
    for frame, label in loader_train:
        print(frame.shape, label.shape)