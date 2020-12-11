import h5py
import numpy as np


class Cache:
    @classmethod
    def get(cls, key):
        raise NotImplemented

    @classmethod
    def set(self, key, value):
        raise NotImplemented


class HDF5Cache(Cache):
    def __init__(self, filename):
        self.filename = filename

    def get(self, key):
        value = None
        with h5py.File(self.filename, 'a') as f:
            if key in f:
                group = f[key]
                data = group['data'][:]
                labels = group['labels'][:]
                value = data, labels
            return value

    def set(self, key, value):
        data, labels = value
        with h5py.File(self.filename, 'a') as f:
            grp = f.create_group(key)
            grp.create_dataset("data", data=data)
            grp.create_dataset("labels", data=labels)

    # @staticmethod
    # def _preprocess_group_value(group):
    #     data = group['data'][:]
    #     labels = group['labels'][:]
    #     result = [(data[i][np.newaxis], labels[i, 0]) for i in range(data.shape[0])]
    #     return result

    # @staticmethod
    # def _preprocess_set_value(value):
    #     data = np.vstack([v[0] for v in value])
    #     labels = np.vstack([v[1] for v in value])
    #     return data, labels
