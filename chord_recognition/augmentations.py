import random

import numpy as np
from torch.utils.data import Dataset


def one_hot(class_ids, num_classes):
    """
    Create one-hot encoding of class ids

    Args:
        class_ids:   ids of classes to map
        num_classes: number of classes

    Returns:
        one-hot encoding of class ids
    """
    oh = np.zeros((len(class_ids), num_classes), dtype=np.float32)
    oh[np.arange(len(class_ids)), class_ids] = 1

    # make sure one-hot encoding corresponds to class ids
    assert (oh.argmax(axis=1) == class_ids).all()
    # make sure there is only one id set per vector
    assert (oh.sum(axis=1) == 1).all()

    return oh


class SemitoneShift(Dataset):
    def __init__(self, dataset_iterator, p, max_shift, bins_per_semitone):
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone
        self._frames = []
        self._init_data(dataset_iterator)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    def _init_data(self, dataset_iterator):
        for data, targets in dataset_iterator:
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
                new_data[i] = np.roll(
                    data[i], shifts[i] * self.bins_per_semitone, axis=-1)

            self._frames.append((new_data, new_targets))

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
        return new_targets.squeeze().astype('long')
