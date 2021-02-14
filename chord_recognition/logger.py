from typing import Iterable, Tuple

from livelossplot import PlotLosses
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class LearningLogger:
    def log_scalars(self, loss: dict):
        raise NotImplemented

    def log_histograms(self, named_parameters: Iterable[Tuple[str, Tensor]]):
        raise NotImplemented


class PlotLossesLogger(LearningLogger, PlotLosses):
    def log_scalars(self, logs, step):
        self.update(logs)
        self.send()

    def log_histograms(self, named_parameters, step):
        pass


class TensorBoardLogger(LearningLogger):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, logs, step):
        for tag, value in logs.items():
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def log_histograms(self, named_parameters, step, bins=1000):
        for tag, value in named_parameters:
            self.writer.add_histogram(
                tag, value.data.cpu().numpy(), step, bins)
            self.writer.add_histogram(
                tag + '/grad', value.grad.data.cpu().numpy(), step, bins)


liveloss = PlotLossesLogger()
