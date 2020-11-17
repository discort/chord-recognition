import random

import numpy as np


def one_hot(class_ids, num_classes):
    """
    Create one-hot encoding of class ids

    Args:
        class_ids:   ids of classes to map
        num_classes: number of classes

    Returns:
        one-hot encoding of class ids
    """
    oh = np.zeros((len(class_ids), num_classes), dtype=np.int64)
    oh[np.arange(len(class_ids)), class_ids] = 1

    # make sure one-hot encoding corresponds to class ids
    assert (oh.argmax(axis=1) == class_ids).all()
    # make sure there is only one id set per vector
    assert (oh.sum(axis=1) == 1).all()

    return oh


class SemitoneShift:
    """
        Augmenter that shifts by semitones a spectrum with logarithmically
        spaced frequency bins.
        :param p: percentage of data to be shifted
        :param max_shift: maximum number of semitones to shift
        :param bins_per_semitone: number of spectrogram bins per semitone
    """

    def __init__(self, p, max_shift, bins_per_semitone):
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone

    def __call__(self, batch_iterator):
        for data, targets in batch_iterator:
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
            yield new_data, new_targets

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
