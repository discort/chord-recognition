from typing import Iterable, Tuple

from livelossplot import PlotLosses
from torch import Tensor


class LearningLogger:
    def log_scalars(self, loss: dict):
        raise NotImplemented

    def log_named_parameters(self, named_parameters: Iterable[Tuple[str, Tensor]]):
        pass


class PlotLossesLogger(PlotLosses):
    def log_scalars(self, logs: dict):
        self.update(logs)
        self.send()


liveloss = PlotLossesLogger()
