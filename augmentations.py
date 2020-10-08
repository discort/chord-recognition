import functools
import random

import numpy as np
import torch


def compose(*functions):
    """Compose a list of function to one."""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions,
                            lambda x: x)


def np_collate(batch):
    """Collate raw numpy data"""
    elem = batch[0]
    elem_type = type(elem)
    if elem_type.__module__ == 'numpy':
        if elem_type.__name__ == 'ndarray':
            return np.stack(batch)
        elif elem.shape == ():  # scalars
            return batch
    elif isinstance(elem, tuple):
        transposed = zip(*batch)
        return [np_collate(samples) for samples in transposed]
    raise TypeError('Invalid input type')


def one_hot(class_ids, num_classes):
    """
    Create one-hot encoding of class ids
    :param class_ids:   ids of classes to map
    :param num_classes: number of classes
    :return: one-hot encoding of class ids
    """
    oh = np.zeros((len(class_ids), num_classes), dtype=np.float32)
    oh[np.arange(len(class_ids)), class_ids] = 1

    # make sure one-hot encoding corresponds to class ids
    assert (oh.argmax(axis=1) == class_ids).all()
    # make sure there is only one id set per vector
    assert (oh.sum(axis=1) == 1).all()

    return oh


class AugmentedIterator:
    """
    Augments (i.e. changes) data and targets of an existing batch iterator
    using a number of augmentation functions.
    Parameters
    ----------
    batch_iterator : Iterator
        Batch iterator to augment
    *augment_fns : callables
        Augmentation functions. They have to accept the values the
        :param:batch_iterator returns, and themselves return similar values.
    Yields
    ------
    tuple of numpy arrays
        Augmented mini-batch of data and targets
    """

    def __init__(self, batch_iterator, *augment_fns):
        self.batch_iterator = batch_iterator
        self.augment = compose(*augment_fns)

    def __iter__(self):
        return self.augment(self.batch_iterator.__iter__())


class IterableSemitoneShift:
    def __init__(self, p, max_shift, bins_per_semitone,
                 target_type='chords_maj_min'):
        """
        Augmenter that shifts by semitones a spectrum with logarithmically
        spaced frequency bins.
        :param p: percentage of data to be shifted
        :param max_shift: maximum number of semitones to shift
        :param bins_per_semitone: number of spectrogram bins per semitone
        :param target_type: specifies target type
        """
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone

        if target_type == 'chords_maj_min':
            self.adapt_targets = self._adapt_targets_chords_maj_min
        elif target_type == 'chroma':
            self.adapt_targets = self._adapt_targets_chroma

    def _adapt_targets_chords_maj_min(self, targets, shifts):
        chord_classes = targets.argmax(-1)
        no_chord_class_index = targets.shape[-1] - 1
        no_chords = (chord_classes == no_chord_class_index)
        chord_roots = chord_classes % 12
        chord_majmin = chord_classes // 12

        new_chord_roots = (chord_roots + shifts) % 12
        new_chord_classes = new_chord_roots + chord_majmin * 12
        new_chord_classes[no_chords] = no_chord_class_index
        new_targets = one_hot(new_chord_classes, no_chord_class_index + 1)
        return new_targets

    def _adapt_targets_chroma(self, targets, shifts):
        new_targets = np.empty_like(targets)
        for i in range(len(targets)):
            new_targets[i] = np.roll(targets[i], shifts[i], axis=-1)
        return new_targets

    def __call__(self, batch_iterator):
        """
        :param batch_iterator: data iterator that yields the data to be
                               augmented
        :return: augmented data/target pairs
        """
        for data, targets in batch_iterator:
            # Temp hack to work with numpy
            data = data.data.numpy()
            targets = targets.data.numpy().astype('long')

            batch_size = len(data)

            shifts = np.random.randint(-self.max_shift,
                                       self.max_shift + 1, batch_size)

            # zero out shifts for 1-p percentage
            no_shift = random.sample(range(batch_size),
                                     int(batch_size * (1 - self.p)))
            shifts[no_shift] = 0

            new_targets = self.adapt_targets(targets, shifts)

            new_data = np.empty_like(data)
            for i in range(batch_size):
                # TODO: remove data from upper and lower parts that got
                #       rolled (?)
                new_data[i] = np.roll(
                    data[i], shifts[i] * self.bins_per_semitone, axis=-1)

            yield torch.from_numpy(new_data), torch.from_numpy(new_targets)


class SemitoneShift:
    def __init__(self, batch_iterator, p, max_shift, bins_per_semitone,
                 target_type='chords_maj_min'):
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone
        self._frames = []
        self._init_data(batch_iterator)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    def _init_data(self, batch_iterator):
        for data, targets in batch_iterator:
            # Temp hack to work with numpy
            data = data.data.numpy()
            targets = targets.data.numpy().astype('long')

            batch_size = len(data)

            shifts = np.random.randint(-self.max_shift,
                                       self.max_shift + 1, batch_size)

            # zero out shifts for 1-p percentage
            no_shift = random.sample(range(batch_size),
                                     int(batch_size * (1 - self.p)))
            shifts[no_shift] = 0

            new_targets = self.adapt_targets(targets, shifts)

            new_data = np.empty_like(data)
            for i in range(batch_size):
                # TODO: remove data from upper and lower parts that got
                #       rolled (?)
                new_data[i] = np.roll(
                    data[i], shifts[i] * self.bins_per_semitone, axis=-1)

            result = (torch.from_numpy(new_data), torch.from_numpy(new_targets))
            self._frames.append(result)

    def adapt_targets(self, targets, shifts):
        chord_classes = targets.argmax(-1)
        no_chord_class_index = targets.shape[-1] - 1
        no_chords = (chord_classes == no_chord_class_index)
        chord_roots = chord_classes % 12
        chord_majmin = chord_classes // 12

        new_chord_roots = (chord_roots + shifts) % 12
        new_chord_classes = new_chord_roots + chord_majmin * 12
        new_chord_classes[no_chords] = no_chord_class_index
        new_targets = one_hot(new_chord_classes, no_chord_class_index + 1)
        return new_targets


class _ChainIterator:
    def __init__(self, loader):
        from itertools import chain
        self._length = self._calculate_length(loader.containers)
        self._iterator = chain(*loader.containers)

    def __iter__(self):
        return self._iterator

    def __len__(self):
        return self._length

    @staticmethod
    def _calculate_length(containers):
        length = 0
        for i in containers:
            length += len(i)
        return length


class ChainLoader:
    """
    Combines a loader and custom iterable.
    Works only for single process
    """

    def __init__(self, *containers):
        self.containers = containers

    def __iter__(self):
        return _ChainIterator(self).__iter__()
