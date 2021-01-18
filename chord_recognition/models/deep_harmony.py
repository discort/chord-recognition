from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from chord_recognition.models.deep_auditory import DeepAuditoryV2


def deep_harmony(
        **kwargs: Any):
    return DeepHarmony(num_classes=26, **kwargs)


class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class DeepHarmony(nn.Module):
    def __init__(self, num_classes: int = 26, n_rnn_layers: int = 1, rnn_dim: int = 128) -> None:
        super(DeepHarmony, self).__init__()
        self.num_classes = num_classes
        self.deep_auditory = DeepAuditoryV2(num_classes=num_classes)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim,
                             dropout=0.5,
                             batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(rnn_dim, self.num_classes)
        )

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
        stack = torch.stack([self.deep_auditory(x[i].view(N, 1, F, S)) for i in range(T)])
        # T x N x F
        stack = stack.transpose(0, 1)
        # N x T x F
        x = self.birnn_layers(stack)
        # N x T x rnn_dim * 2
        x = self.classifier(x)
        # N x T x C
        x = x.transpose(0, 1)
        # T x N x C
        x = x.log_softmax(2)
        # T x N x C
        return x
