import os
import os.path
import random

import librosa
from imblearn.datasets import make_imbalance
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler

from .ann_utils import convert_chord_ann_matrix, get_chord_labels
from .utils import read_audio, context_window, log_filtered_spectrogram

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def undersample_dataset(dataset, sampling_strategy, random_state):
    """
    Balance imbalanced dataset using sampling strategy
    """
    y = [yi for _, yi in dataset]
    # reshape X to have (n_samples, n_features)
    X = np.vstack([x.reshape(1, -1) for x, _ in dataset])
    X, y = make_imbalance(
        X, y, sampling_strategy=sampling_strategy,
        random_state=random_state)
    # Reshape X to it's original form
    X = [X[i].reshape(1, 105, 15) for i in range(X.shape[0])]
    return X, y


def split_datasource(datasource, lengths):
    """
    Split a datasource into non-overlapping new datasources of given lengths
    """
    from torch import randperm
    from torch._utils import _accumulate
    if sum(lengths) != len(datasource):
        raise ValueError("Sum of input lengths does not equal the length of the input datasource")

    indices = randperm(sum(lengths)).tolist()
    return [[datasource[i] for i in indices[offset - length: offset]]
            for offset, length in zip(_accumulate(lengths), lengths)]


def prepare_datasource(
        datasets,
        data_dir=None,
        excluded_files=(),
        allowed_files=()):
    """
    Prepares data for Pytorch Dataset.
    Walks through each dir dataset and collects labeled and source files.

    Args:
        datasets:iterable - dir names that contain labels and source files

    Returns:
        result: list of tuples in [(path_to_label:path_to_source)] notation
    """
    datasource = []
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, 'data')
    for ds_name in datasets:
        data_source_dir = os.path.join(data_dir, ds_name)
        lab_files = collect_files(
            dir_path=os.path.join(data_source_dir, 'chordlabs'),
            ext='.lab',
            excluded_files=excluded_files,
            allowed_files=allowed_files)
        audio_files = collect_files(
            dir_path=os.path.join(data_source_dir, 'mp3'),
            ext='.mp3',
            excluded_files=excluded_files,
            allowed_files=allowed_files)
        if len(lab_files) != len(audio_files):
            raise ValueError(f"{ds_name} has different len of lab and mp3 files")

        for lab, audio in zip(lab_files, audio_files):
            assert lab.split('/')[-1].replace('.lab', '').lower() ==\
                audio.split('/')[-1].replace('.mp3', '').lower()
            datasource.append((lab, audio))
    return datasource


def collect_files(dir_path, ext='.lab', excluded_files=(), allowed_files=()):
    files = []
    for dirname in sorted(os.listdir(dir_path)):
        for filename in sorted(os.listdir(os.path.join(dir_path, dirname))):
            if allowed_files and not any(f in filename for f in allowed_files):
                continue

            if any(f in filename for f in excluded_files):
                continue
            if not filename.endswith(ext):
                continue

            file_path = os.path.join(dir_path, dirname, filename)
            files.append(file_path)
    return files


class ChromaDataset:
    def __init__(self, datasource, window_size=4096, hop_length=2048,
                 context_size=7, cache=None, transform=None):
        """
        Args:
            datasource: list of tuples - label:source file path notation
            cache: cache.Cache obj - use cached data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datasource = datasource
        self.window_size = window_size
        self.hop_length = hop_length
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)
        self.num_classes = len(self.chord_labels)
        self._frames = []
        self.context_size = context_size
        self.cache = cache
        self.transform = transform
        self._init_dataset()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._frames[idx]
        elif isinstance(idx, np.ndarray):
            frames = np.array(self._frames)[idx]
            return [(f[0], f[1]) for f in frames]

    def _init_dataset(self):
        """
        Initialise (sample, label) for all frames.
        Use cache if available.
        """
        for ann_path, audio_path in self.datasource:
            sample = self._make_sample(ann_path, audio_path)
            if self.context_size is None:
                frame_data = [sample]
            else:
                frame_data = self._make_frames(sample)
            self._frames.extend(frame_data)

    def _make_frames(self, sample):
        spec, ann_matrix = sample
        result = []

        if self.context_size:
            container = context_window(spec, self.context_size)
        else:
            # default container w/o context
            container = (spec[:, i] for i in range(spec.shape[1]))

        # Initialize sample/target pairs
        for idx, frame in enumerate(container):
            label = ann_matrix[:, idx]
            # Exclude only unlabeled data
            if not label.any():
                continue

            if self.transform:
                frame = self.transform(frame)
            result.append((frame.reshape(1, *frame.shape), np.argmax(label)))
        return result

    def _cached(func):
        def wrapped(self, *args):
            value = None
            key = self._prepare_key(args[0])
            cache = self.cache

            if cache:
                value = cache.get(key)
            if not value:
                value = func(self, *args)
                if cache:
                    cache.set(key, value)
            return value
        return wrapped

    @staticmethod
    def _prepare_key(key):
        # Make key in <album_annotation> format
        return '_'.join(key.split('/')[-2:]).replace('.lab', '')

    @_cached
    def _make_sample(self, ann_path, audio_path):
        audio_waveform, sampling_rate = read_audio(audio_path, sr=None, mono=True)

        spec = log_filtered_spectrogram(
            audio_waveform=audio_waveform,
            sr=sampling_rate,
            window_size=self.window_size,
            hop_length=self.hop_length,
            fmin=65, fmax=2100, num_bands=24)

        Fs_X = sampling_rate / self.hop_length
        ann_matrix, _, _, _, _ = convert_chord_ann_matrix(
            fn_ann=ann_path, chord_labels=self.chord_labels,
            Fs=Fs_X, N=spec.shape[1], last=False)

        return spec, ann_matrix
