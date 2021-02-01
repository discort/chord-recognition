from collections import OrderedDict
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from chord_recognition.models.deep_auditory import DeepAuditoryV2
from chord_recognition.utils import ctc_greedy_decoder, expand_labels

from .layers import script_lnlstm, LSTMState

CURR_DIR = os.path.dirname(__file__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def deep_harmony(pretrained: bool = False,
                 model_name: str = 'deep_harmony_exp6.pth',
                 **kwargs: Any):
    if pretrained:
        model = DeepHarmony(**kwargs)
        state_dict = torch.load(
            os.path.join(CURR_DIR, model_name),
            map_location=device)
        model.load_state_dict(state_dict)
        return model
    return DeepHarmony(**kwargs)


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
                 input_size: int,
                 hidden_size: int,
                 rnn_type: nn.RNNBase = nn.LSTM,
                 bidirectional: bool = False,
                 batch_norm: bool = False):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, h = self.rnn(x)
        if self.bidirectional:
            # (TxNxH*2) -> (TxNxH) by sum
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int,
                 stride: int,
                 dropout: float,
                 n_feats: int,
                 nonlinearity: nn.Module = nn.GELU,
                 downsample=None):
        super(ResidualCNN, self).__init__()
        self.nonlinearity = nonlinearity()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2, bias=False)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = self.nonlinearity(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        return x  # (batch, channel, feature, time)


class ResChroma(nn.Module):
    def __init__(self,
                 n_feats: int,
                 n_cnn_layers: int = 3,
                 out_dim: int = 128,
                 block_depth: int = 1,
                 dropout: float = 0.1,
                 nonlinearity: nn.Module = nn.GELU) -> None:
        super(ResChroma, self).__init__()
        assert 0 < block_depth < 3

        n_feats = n_feats // 2 + 1
        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        self.cnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1,
                        dropout=dropout, n_feats=n_feats, nonlinearity=nonlinearity)
            for _ in range(n_cnn_layers)
        ])
        fc_input = n_feats * 32
        if block_depth == 2:
            self.cnn_layers = nn.Sequential(
                self.cnn_layers,
                self._make_layer(32, 64, n_cnn_layers, dropout, n_feats, nonlinearity)
            )
            fc_input = n_feats * 64

        self.norm = CNNLayerNorm(n_feats)
        self.fully_connected = nn.Linear(fc_input, out_dim, bias=False)

    def _make_layer(self,
                    in_planes,
                    out_planes,
                    n_cnn_layers,
                    dropout,
                    n_feats,
                    nonlinearity):
        layers = []
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        conv = ResidualCNN(in_planes, out_planes,
                           kernel=3,
                           stride=1,
                           dropout=dropout,
                           n_feats=n_feats,
                           nonlinearity=nonlinearity,
                           downsample=downsample)
        layers.append(conv)
        for _ in range(n_cnn_layers - 1):
            conv = ResidualCNN(out_planes, out_planes,
                               kernel=3,
                               stride=1,
                               dropout=dropout,
                               n_feats=n_feats,
                               nonlinearity=nonlinearity)
            layers.append(conv)
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, x: Tensor) -> Tensor:
        """
        N - batch size
        F - features
        T - time
        """
        # N x 1 x F x T
        x = self.cnn(x)
        # N x 32 x F x T
        x = self.cnn_layers(x)
        # N x 32 x F x T
        x = self.norm(x)
        # N x 32 x F x T
        x = F.relu(x)
        # N x 32 x F x T
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        # N x F x T
        x = x.transpose(1, 2)  # (batch, time, feature)
        # N x T x F
        x = self.fully_connected(x)
        # N x T x F
        x = x.transpose(0, 1)
        # T x N x F
        return x


class DeepHarmony(nn.Module):
    def __init__(self,
                 n_feats: int,
                 rnn_type: nn.RNNBase = nn.LSTM,
                 num_classes: int = 26,
                 n_rnn_layers: int = 5,
                 rnn_dim: int = 128,
                 rnn_hidden_size: int = 128,
                 cnn_kwargs: dict = {},
                 bidirectional: bool = False) -> None:
        super(DeepHarmony, self).__init__()
        self.num_classes = num_classes
        # #self.cnn_layers = DeepAuditoryV2(num_classes=num_classes)
        cnn_kwargs.setdefault('n_cnn_layers', 3)
        cnn_kwargs.setdefault('out_dim', rnn_dim)
        self.cnn = ResChroma(n_feats=n_feats,
                             **cnn_kwargs)
        self.rnn_dim = rnn_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.n_rnn_layers = n_rnn_layers
        self.decoder = ctc_greedy_decoder
        self.rnn_layers = script_lnlstm(input_size=rnn_dim,
                                        hidden_size=rnn_hidden_size,
                                        num_layers=n_rnn_layers,
                                        bidirectional=False)
        fully_connected = nn.Sequential(
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, self.num_classes, bias=False)
        )
        self.fc = nn.Sequential(SequenceWise(fully_connected))

    def forward(self, x: Tensor) -> Tensor:
        """
        x - has shape [batch_size x channel x n_feats x input_length)
            (N x 1 x F x T)
        C - number of classes
        """
        N, _, _, _ = x.shape
        # N x 1 x F x T
        x = self.cnn(x)
        # T x N x F
        states = [LSTMState(torch.zeros(N, self.rnn_hidden_size, dtype=x.dtype, device=x.device),
                            torch.zeros(N, self.rnn_hidden_size, dtype=x.dtype, device=x.device))
                  for _ in range(self.n_rnn_layers)]
        x, _ = self.rnn_layers(x, states)
        # T x N x rnn_hidden_size
        x = self.fc(x)
        # T x N x C
        return x

    @torch.no_grad()
    def predict(self, dataloader):
        """
        Pass input data through the model sequentually

        Args:
            dataloader (torch.DataLoader) - input data

        Returns:
            result - [(input_size, out_labels)]
        """
        pred_labels = []  # (seq_len, [batch_labels])
        for inputs in dataloader:
            # Convert to (batch, channel, features, time)
            inputs = inputs.unsqueeze(1).transpose(2, 3)
            out = self.forward(inputs).data.numpy()  # T x N x C
            decoded_out = self.decoder(out)

            for batch in decoded_out:
                pred_labels.append((inputs.shape[-1], batch))

        result = self._match_labels_frames(pred_labels)
        return result

    @staticmethod
    def _match_labels_frames(pred_labels):
        """
        Match predicted labels to initial input provided by the number of frames

        Args:
            pred_labels ([tuple]) - predicted labels in format (seq_len, out_labels)

        Returns:
            np.array - result with labels matched initial input
        """
        result = expand_labels(pred_labels)
        result = np.array(result, dtype=np.int32)
        return result
