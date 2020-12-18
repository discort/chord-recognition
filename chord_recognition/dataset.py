import os
import os.path
import random

import librosa
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler

from .ann_utils import convert_chord_ann_matrix, get_chord_labels
from .utils import compute_chromagram, read_audio, scale_data, log_filtered_spectrogram,\
    preprocess_spectrogram

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


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
            if not self.context_size:
                frame_data = [sample]
            else:
                frame_data = self._make_frames(sample)
            self._frames.extend(frame_data)

    def _make_frames(self, sample):
        spec, ann_matrix = sample
        result = []

        # Context window cannot be applied because the length of frames is too short
        # if self.context_size:
        #     if spec.shape[1] < (2 * self.context_size + 1):
        #         return result

        container = context_window(spec, self.context_size)
        for frame, idx_target in zip(container, range(spec.shape[1])):
            label = ann_matrix[:, idx_target]
            # Exclude only unlabeled data
            if not label.any():
                continue

            if self.transform:
                frame = self.transform(frame)
            result.append((frame.reshape(1, *frame.shape), np.argmax(label)))
        return result

    def _preprocess_sample(self, sample):
        spec, ann_matrix = sample
        return spec, ann_matrix

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
        audio_waveform, sampling_rate = read_audio(audio_path, Fs=None, mono=True)
        # spec = compute_chromagram(audio_waveform=audio_waveform,
        #                           Fs=sampling_rate,
        #                           n_chroma=105,
        #                           window_size=self.window_size,
        #                           hop_length=self.hop_length)
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


excluded_files = (
    # zweieck
    '09_-_Mr_Morgan',
    '01_-_Spiel_Mir_Eine_Alte_Melodie',
    '11_-_Ich_Kann_Heute_Nicht',
    # queen
    '14 Hammer To Fall',
    '08 Save Me',
    # robbie_williams
    '11-Man Machine',
    '01-Ghosts',
    '11-A Place To Crash',
    '08-Heaven From Here',
    '09-Random Acts Of Kindness',
    '05-South Of The Border',
)
ds = prepare_datasource(
    ('zweieck', 'queen', 'robbie_williams'),
    excluded_files=excluded_files)

allowed_files = (
    '06-Mr_Moonlight',
    '06-Yellow_Submarine',
    '03-I_m_Only_Sleeping',
    '09-Penny_Lane',
    '12-Wait',
    '11-Do_You_Want_To_Know_A_Secret',
    '12-A_Taste_Of_Honey',
    '04-I_m_Happy_Just_To_Dance_With_You',
    '03-If_I_Fell',
    '10-I_m_Looking_Through_You',
    '09-When_I_m_Sixty-Four',
    '06-Till_There_Was_You',
    '05-Octopus_s_Garden',
    '03-All_My_Loving',
    '05-And_I_Love_Her',
    '02-All_I_ve_Got_To_Do',
    '10-For_No_One',
    '08-Because',
    '06-She_s_Leaving_Home',
    '04-Chains',
    '10-Things_We_Said_Today',
    '09-One_After_909',
    '09-Girl',
    '14-Run_For_Your_Life',
    '04-Oh_Darling',
    '04-Don_t_Bother_Me',
    '06-I_Want_You_She_s_So_Heavy_',
    '06-Tell_Me_Why',
)
beatles_ds = prepare_datasource(('beatles',), allowed_files=allowed_files)
balanced_datasource = ds + beatles_ds
