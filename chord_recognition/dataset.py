import os
import os.path
import random

import librosa
from imblearn.datasets import make_imbalance
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler

from .ann_utils import convert_chord_ann_matrix, get_chord_labels
from .utils import read_audio, split_with_context, stack_frames, log_filtered_spectrogram

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


class SpecDataset:
    def __init__(
            self,
            datasource,
            window_size=4096,
            hop_length=4410,
            cache=None):
        """
        Args:
            datasource: list[tuple] - label:source file path location
            window_size (int)
            hop_length (int)
            cache: cache.Cache obj - use cached data
        """
        self.datasource = datasource
        self.window_size = window_size
        self.hop_length = hop_length
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)
        self.num_classes = len(self.chord_labels)
        self._data = []
        self.cache = cache
        self._init_dataset()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        inputs, targets = self._data[idx]
        return inputs, targets

    def _init_dataset(self):
        """
        Initialise (sample, label) for all frames.
        Use cache if available.
        """
        for ann_path, audio_path in self.datasource:
            sample = self._make_sample(ann_path, audio_path)
            self._data.append(sample)

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
        # ToDo:
        # check audio sample-rate to equal dataset sample rate
        audio_waveform, sampling_rate = read_audio(audio_path, sr=None, mono=True)

        spec = log_filtered_spectrogram(
            audio_waveform=audio_waveform,
            sr=sampling_rate,
            window_size=self.window_size,
            hop_length=self.hop_length,
            fmin=65, fmax=2100, num_bands=24)

        fps = sampling_rate / self.hop_length
        ann_matrix, _, _, _, _ = convert_chord_ann_matrix(
            fn_ann=ann_path, chord_labels=self.chord_labels,
            Fs=fps, N=spec.shape[1], last=False)

        return spec, ann_matrix


class FrameDataset:
    def __init__(
            self,
            dataset,
            context_size=7,
            transform=None):
        """
        Args:
            context_size (int)
            transform (callable): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        self._frames = []
        self.context_size = context_size
        self.transform = transform
        self._init_dataset()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int64)):
            inputs, labels = self._frames[idx]
            if self.transform:
                inputs = self.transform(inputs)
            return inputs, labels
        elif isinstance(idx, np.ndarray):
            frames = np.array(self._frames)[idx]
            return [(f[0], f[1]) for f in frames]

    def _init_dataset(self):
        """
        Initialise (sample, label) for all frames.
        Use cache if available.
        """
        for sample in self.dataset:
            frame_data = self._make_frames(sample)
            self._frames.extend(frame_data)

    def _make_frames(self, sample):
        spec, ann_matrix = sample
        result = []

        container = split_with_context(spec, self.context_size)

        # Initialize input/target pairs
        for idx, frame in enumerate(container):
            label = ann_matrix[:, idx]
            # Exclude only unlabeled data
            if not label.any():
                continue

            if self.transform:
                frame = self.transform(frame)
            result.append((frame.reshape(1, *frame.shape), np.argmax(label)))
        return result


class SequenceFrameDataset(FrameDataset):
    def __init__(
            self,
            dataset,
            context_size=7,
            seq_length=20,
            target_length=8,
            transform=None):
        """
        Args:
            seq_length (int) - sequence length in frames
            target_length (int) - max target length
        """
        self.dataset = dataset
        self.seq_length = seq_length
        self.target_length = target_length
        self.context_size = context_size
        self._frames = []
        self.transform = transform
        self._init_dataset()

    def _init_dataset(self):
        for sample in self.dataset:
            frame_data = self._make_frames(sample)
            self._frames.extend(frame_data)

    def _make_frames(self, sample):
        spec, ann_matrix = sample
        result = []

        # Exclude short input lengths to avoid negative log likelihood (that is ctc loss is)
        # https://discuss.pytorch.org/t/ctc-loss-with-variable-input-lengths-produces-nan-values/43476/5
        spec_frames = stack_frames(spec.T, self.seq_length, last=False)
        label_frames = np.argmax(ann_matrix, 0)
        label_frames = stack_frames(label_frames, self.seq_length, last=False)
        # Initialize input/target pairs
        for spec_frame, label_frame in zip(spec_frames, label_frames):
            # Exclude blank labeled data
            if not label_frame.any():
                continue

            label_frame = self._cleanup_labels(label_frame)
            result.append((spec_frame, label_frame))
        return result

    def _cleanup_labels(self, labels):
        """
        Sequentially remove duplicate labels
        """
        N = len(labels)
        result = []
        prev_label = None
        blank_index = 0
        for i in range(N):
            if labels[i] == blank_index:
                continue

            if prev_label is None:
                prev_label = labels[i]
                result.append(prev_label)

            if prev_label != labels[i]:
                result.append(labels[i])
                prev_label = labels[i]

        result = np.array(result)
        # offset = self.target_length - len(result)
        #result = np.pad(result, pad_width=(0, offset), mode='constant', constant_values=0)
        # Make targets shorter than inputs. Avoid infinite losses
        return result
