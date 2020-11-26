import os
import os.path
import random

import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from .utils import convert_chord_ann_matrix, get_chord_labels, read_structure_annotation,\
    convert_chord_label, convert_ann_to_seq_label, compute_chromagram, read_audio, log_compression

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


def flatten_iterator(data_source):
    result = []
    for inputs, targets in data_source:
        batch_size = inputs.shape[0]
        data = zip(np.array_split(inputs, batch_size), np.array_split(targets, batch_size))
        result.extend(data)
    return [(i.squeeze(0), t.squeeze(0)) for i, t in result]


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


def prepare_datasource(datasets):
    """
    Prepares data for Pytorch Dataset.
    Walks through each dir dataset and collects labeled and source files.

    Args:
        datasets:iterable - dir names that contain labels and source files

    Returns:
        result: list of tuples in [(path_to_label:path_to_source)] notation
    """
    datasource = []
    data_dir = os.path.join(BASE_DIR, 'data')
    for ds_name in datasets:
        data_source_dir = os.path.join(data_dir, ds_name)
        lab_files = collect_files(dir_path=os.path.join(data_source_dir, 'chordlabs'), ext='.lab')
        audio_files = collect_files(dir_path=os.path.join(data_source_dir, 'mp3'), ext='.mp3')
        if len(lab_files) != len(audio_files):
            raise ValueError(f"{ds_name} has different len of lab and mp3 files")

        lab_audio = [(lab, audio) for lab, audio in zip(lab_files, audio_files)]
        datasource.extend(lab_audio)
    return datasource


def collect_files(dir_path, ext='.lab', excluded_files=()):
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        for filename in filenames:
            if any(f in filename for f in excluded_files):
                continue
            if not filename.endswith(ext):
                continue
            file_path = os.path.join(root, filename)
            files.append(file_path)
    return files


def iterate_batches(data_source, batch_size, shuffle=False, expand=True):
    """
    Generates mini-batches from a data source.
    Args:
        data_source - Data source to generate mini-batches from
        batch_size : int - Number of data points and targets in each mini-batch
        shuffle : bool - Indicates whether to randomize the items in each mini-batch
    expand : bool
        Indicates whether to fill up the last mini-batch with
        random data points if there is not enough data available.
    Returns:
        mini-batch of data and targets
    """

    idxs = list(range(len(data_source)))

    if shuffle:
        random.shuffle(idxs)

    start_idx = 0
    while start_idx < len(data_source):
        batch_idxs = idxs[start_idx:start_idx + batch_size]

        # last batch could be too small
        if len(batch_idxs) < batch_size and expand:
            # fill up with random indices not yet in the set
            n_missing = batch_size - len(batch_idxs)
            batch_idxs += random.sample(idxs[:start_idx], n_missing)

        start_idx += batch_size
        result_inputs = []
        result_targets = []
        for i in batch_idxs:
            inputs, targets = data_source[i]
            result_inputs.append(inputs)
            result_targets.append(targets)
        yield np.stack(result_inputs), np.stack(result_targets)
        # Concatdataset does not support list indexing
        #yield data_source[batch_idxs]


class BatchIterator:
    """
    Iterates over mini batches of a data source.
    """

    def __init__(self, data_source, batch_size, shuffle=False, expand=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.expand = expand

    def __iter__(self):
        return iterate_batches(self.data_source, self.batch_size, self.shuffle, self.expand)


class ContextIterator:
    """Allows iterate through the data with context

    X_i = [l_s, ..., l_i, ..., lt]
    s = i - C
    t = i + C
    i - index of the target
    C - context size
    """

    def __init__(self, data, context_size):
        self.data = data
        self.context_size = context_size

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        n = self.data.shape[1]
        if self.index >= n:
            raise StopIteration

        if self.index < self.context_size:
            start = 0
            end = 2 * self.context_size + 1
        elif (self.index + self.context_size) >= n:
            start = n - (2 * self.context_size) - 1
            end = n
        else:
            start = self.index - self.context_size
            end = self.index + self.context_size + 1
        self.index += 1
        return self.data[:, start:end]


def context_window(x, context_size):
    assert context_size % 2 == 1, "context_size must be odd"
    assert x.shape[1] > 2 * context_size + 1, "Low size x"
    pad_number = context_size
    left_pad_idx = pad_number
    M, N = x.shape
    dtype = x.dtype
    #result = np.zeros((N, M, 2 * context_size + 1))
    result = []
    right_pad_idx = 1
    for i in range(N):
        if i - pad_number < 0:
            right = x[:, range(0, i + pad_number + 1)]
            left_pad = x[:, :left_pad_idx]
            window = np.concatenate([left_pad, right], axis=1)
            left_pad_idx -= 1
        elif i + pad_number >= N:
            left = x[:, range(i - pad_number, N)]
            right_pad = x[:, -right_pad_idx:]
            window = np.concatenate([left, right_pad], axis=1)
            right_pad_idx += 1
        else:
            indexes = list(range(i - pad_number, i)) + list(range(i, i + pad_number + 1))
            window = x[:, indexes]
        #result[i, :, :] = window
        result.append(window.astype(dtype))
    return result


class MirDataset(Dataset):
    def __len__(self):
        raise NotImplemented

    def __getitem__(self, idx):
        raise NotImplemented

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


class AudioDataset(MirDataset):
    def __init__(self, audio_dir, ann_dir, window_size=4096, hop_length=2048):
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
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.ann_list[idx] + '.mp3')
        audio_waveform, Fs = read_audio(audio_path, Fs=None, mono=True)

        chromagram = compute_chromagram(audio_waveform=audio_waveform,
                                        Fs=Fs,
                                        window_size=self.window_size,
                                        hop_length=self.hop_length)
        chromagram = log_compression(chromagram)
        N_X = chromagram.shape[1]
        Fs_X = Fs / self.hop_length

        ann_path = os.path.join(self.ann_dir, self.ann_list[idx] + '.lab')
        ann_matrix, _, _, _, ann_seg_sec = convert_chord_ann_matrix(
            ann_path, self.chord_labels, Fs=Fs_X, N=N_X, last=False)

        sample = {
            'sample': self.ann_list[idx],
            'audio_waveform': audio_waveform,
            'Fs': Fs,
            'chromagram': chromagram,
            'ann_matrix': ann_matrix,
            'ann_seg_sec': ann_seg_sec,
        }
        return sample


class ChromaDataset(MirDataset):
    def __init__(self, datasource, window_size=4096,
                 hop_length=2048, n_chroma=105, transform=None):
        """
        Args:
            datasource: list of tuples - label:source file path notation
            audio_dir (string): Path to audio dir
            ann_dir (string): Path to the dir with csv annotations.
            n_chroma: int - number of chroma features
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.audio_dir = audio_dir
        # self.ann_dir = ann_dir
        self.datasource = datasource
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        #self.ann_list = self._build_ann_list()
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)
        self.num_classes = len(self.chord_labels)
        self._frames = []
        self.context_size = 7
        self.transform = transform
        self._init_dataset()

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._frames[idx]
        elif isinstance(idx, list):
            frame = self._frames[0]
            batch_size = len(idx)
            result_inputs = np.empty((batch_size, *frame[0].shape))
            result_targets = np.empty((batch_size, *frame[1].shape), dtype=np.int64)
            for i, batch_idx in enumerate(idx):
                result_inputs[i, :] = self._frames[batch_idx][0]
                result_targets[i, :] = self._frames[batch_idx][1]
            return result_inputs, result_targets

    def _init_dataset(self):
        for ann_path, audio_path in self.datasource:
            audio_frames = self._init_audio(ann_path, audio_path)
            self._frames.extend(audio_frames)

    def _init_audio(self, ann_path, audio_path):
        #audio_path = os.path.join(self.audio_dir, filename + '.mp3')
        audio_waveform, sampling_rate = read_audio(audio_path, Fs=None, mono=True)

        chromagram = compute_chromagram(audio_waveform=audio_waveform,
                                        Fs=sampling_rate,
                                        n_chroma=self.n_chroma,
                                        window_size=self.window_size,
                                        hop_length=self.hop_length)
        chromagram = log_compression(chromagram)
        N_X = chromagram.shape[1]
        Fs_X = sampling_rate / self.hop_length

        # ann_path = os.path.join(self.ann_dir, filename + '.lab')
        ann_matrix, _, _, _, _ = convert_chord_ann_matrix(
            fn_ann=ann_path, chord_labels=self.chord_labels, Fs=Fs_X, N=N_X, last=False)

        # Exclude unlabeled data
        zero_indices = np.all(ann_matrix == 0, axis=0)
        if zero_indices.any():
            chromagram = chromagram[:, ~zero_indices]
            ann_matrix = ann_matrix[:, ~zero_indices]

        if self.transform:
            chromagram, ann_matrix = self.transform((chromagram.T, ann_matrix.T))
            chromagram, ann_matrix = chromagram.T, ann_matrix.T

        result = []
        container = context_window(chromagram, self.context_size)
        for frame, idx_target in zip(container, range(chromagram.shape[1])):
            label = ann_matrix[:, idx_target]
            result.append((frame.reshape(1, *frame.shape), label))
        return result


class FrameLabelDataset(MirDataset):
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
        _, sampling_rate = read_audio(audio_path, Fs=None, mono=True)
        Fs_X = sampling_rate / self.hop_length

        ann_path = os.path.join(self.ann_dir, filename + '.lab')
        ann_seg_ind = read_structure_annotation(ann_path, Fs=Fs_X, index=True)
        ann_seg_ind = convert_chord_label(ann_seg_ind)
        result = convert_ann_to_seq_label(ann_seg_ind)
        return result
