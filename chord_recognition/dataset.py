import os
import os.path
import random

import numpy as np
from torch.utils.data import Dataset

from .utils import convert_chord_ann_matrix, get_chord_labels, read_structure_annotation,\
    convert_chord_label, convert_ann_to_seq_label, compute_chromagram, read_audio, log_compression


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
        yield data_source[batch_idxs]


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
    def __init__(self, audio_dir, ann_dir, window_size=4096,
                 hop_length=2048, n_chroma=105):
        """
        Args:
            audio_dir (string): Path to audio dir
            ann_dir (string): Path to the dir with csv annotations.
            n_chroma: int - number of chroma features
        """
        self.audio_dir = audio_dir
        self.ann_dir = ann_dir
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.ann_list = self._build_ann_list()
        self.chord_labels = get_chord_labels(ext_minor='m', nonchord=True)
        self.num_classes = len(self.chord_labels)
        self._frames = []
        self.context_size = 7
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
            result_targets = np.empty((batch_size, *frame[1].shape))
            for i, batch_idx in enumerate(idx):
                result_inputs[i, :] = self._frames[batch_idx][0]
                result_targets[i, :] = self._frames[batch_idx][1]
            return result_inputs, result_targets

    def _init_dataset(self):
        for filename in self.ann_list:
            audio_frames = self._init_audio(filename)
            self._frames.extend(audio_frames)

    def _init_audio(self, filename):
        audio_path = os.path.join(self.audio_dir, filename + '.mp3')
        audio_waveform, sampling_rate = read_audio(audio_path, Fs=None, mono=True)

        chromagram = compute_chromagram(audio_waveform=audio_waveform,
                                        Fs=sampling_rate,
                                        n_chroma=self.n_chroma,
                                        window_size=self.window_size,
                                        hop_length=self.hop_length)
        chromagram = log_compression(chromagram)
        N_X = chromagram.shape[1]
        Fs_X = sampling_rate / self.hop_length

        ann_path = os.path.join(self.ann_dir, filename + '.lab')
        ann_matrix, _, _, _, _ = convert_chord_ann_matrix(
            fn_ann=ann_path, chord_labels=self.chord_labels, Fs=Fs_X, N=N_X, last=False)

        result = []
        #container = context_window(chromagram, self.context_size)
        container = ContextIterator(chromagram, self.context_size)
        for frame, idx_target in zip(container, range(N_X)):
            #label = np.argmax(ann_matrix[:, idx_target])
            label = ann_matrix[:, idx_target]
            if not np.any(label):  # Exclude unlabeled data
                continue
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
