from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from chord_recognition.models.deep_auditory import DeepAuditoryV2


def deep_harmony(
        **kwargs: Any):
    return DeepHarmony(num_classes=26, **kwargs)


class DeepHarmony(nn.Module):
    def __init__(self, num_classes: int = 26) -> None:
        super(DeepHarmony, self).__init__()
        self.num_classes = num_classes
        self.deep_auditory = DeepAuditoryV2(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        N, T, F, S = x.shape
        N - batch size
        T - amount of stacked frames
        F - features
        S - frame size
        C - number of classes
        """
        N, T, F, S = x.shape
        # N x T x F x S
        x = x.transpose(0, 1)
        # T x N x F x S
        stack = torch.empty(T, N, self.num_classes)
        for i in range(S):
            stack[i] = self.deep_auditory(x[i].view(N, 1, F, S))  # N x C
        # T x N x C
        x = stack.log_softmax(2)
        # T x N x C
        return x
