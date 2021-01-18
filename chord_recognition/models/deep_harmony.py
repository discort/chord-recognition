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


# Taken from  https://github.com/SeanNaren/deepspeech.pytorch with modifications
class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x


class BatchRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_type=nn.LSTM,
                 bidirectional=False,
                 batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, h = self.rnn(x)
        return x


class DeepHarmony(nn.Module):
    def __init__(self,
                 num_classes: int = 26,
                 n_rnn_layers: int = 3,
                 rnn_dim: int = 128) -> None:
        super(DeepHarmony, self).__init__()
        self.num_classes = num_classes
        self.cnn_layers = DeepAuditoryV2(num_classes=num_classes)
        self.rnn_layers = nn.Sequential(*[
            BatchRNN(input_size=rnn_dim,
                     hidden_size=rnn_dim)
            for i in range(n_rnn_layers)
        ])
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_dim),
            nn.Linear(rnn_dim, self.num_classes, bias=False)
        )
        self.fc = nn.Sequential(SequenceWise(fully_connected))

    def forward(self, x: Tensor) -> Tensor:
        """
        N, T, F, S = x.shape
        N - batch size
        T - amount of stacked frames
        F_in - input features
        F - produced features
        S - frame size
        C - number of classes
        """
        N, T, F, S = x.shape
        # N x T x F_in x S
        x = x.transpose(0, 1)
        # T x N x F_in x S
        x = torch.stack([self.cnn_layers(x[i].view(N, 1, F, S)) for i in range(T)])
        # T x N x F
        x = self.rnn_layers(x)
        # T x N x rnn_dim
        x = self.fc(x)
        # T x N x C
        x = x.log_softmax(2)
        # T x N x C
        return x
