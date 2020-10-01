import torch


class ContextContainer:
    def __init__(self, data, context_size):
        self.data = data
        self.context_size = context_size

    def __iter__(self):
        self.index = 0
        return self

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


def main():
    spectrogram = torch.randn(105, 855)
    container = ContextContainer(spectrogram, 7)
    for i, x in enumerate(container):
        print(x.shape, i)


if __name__ == '__main__':
    main()
