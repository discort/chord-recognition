from collections import OrderedDict
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

CURR_DIR = os.path.dirname(__file__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def deep_auditory(pretrained: bool = False, **kwargs: Any):
    if pretrained:
        model = DeepAuditory(**kwargs)
        state_dict = torch.load(
            os.path.join(CURR_DIR, 'models/best_model.pt'),
            map_location=device)
        model.load_state_dict(state_dict)
        return model
    return DeepAuditory(**kwargs)


def fact_auditory(**kwargs: Any):
    return FactorizationAuditory(**kwargs)


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        # 1. In original paper, <https://arxiv.org/abs/1502.03167>
        # activation should be applied after batchnorm
        # however, it's arguable what to use first: batchnorm or activation
        # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
        # 2. This function converges better if relu is used before batchnorm
        x = F.relu(x)
        x = self.bn(x)
        return x


class DeepAuditory(nn.Module):
    """Deep Auditory Model architecture from
    'A Fully Convolutional Deep Auditory Model for Musical Chord Recognition'
    <https://arxiv.org/abs/1612.05082>
    """

    def __init__(self, num_classes: int = 25) -> None:
        super(DeepAuditory, self).__init__()

        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.Conv2d_3a_3x3 = BasicConv2d(32, 64, kernel_size=3)
        self.Conv2d_4a_3x3 = BasicConv2d(64, 64, kernel_size=3)
        self.Conv2d_5a_12x9 = BasicConv2d(64, 128, kernel_size=(12, 9))
        self.Conv2d_6a_1x1 = BasicConv2d(128, 25, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(13, 3))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: Tensor) -> Tensor:
        # N x 1 x 105 x 15
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 105 x 15
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 105 x 15
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 105 x 15
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 105 x 15
        x = self.maxpool1(x)
        # N x 32 × 52 × 15
        x = self.dropout(x)
        # N x 32 × 52 × 15
        x = self.Conv2d_3a_3x3(x)
        # N x 64 × 50 × 13
        x = self.Conv2d_4a_3x3(x)
        # N x 64 × 48 × 11
        x = self.maxpool1(x)
        # N x 64 × 24 × 11
        x = self.dropout(x)
        # N x 64 × 24 × 11
        x = self.Conv2d_5a_12x9(x)
        # N x 128 × 13 × 3
        x = self.dropout(x)
        # N x 128 × 13 × 3
        x = self.Conv2d_6a_1x1(x)
        # N x 25 × 13 × 3
        x = self.avg_pool(x)
        # N x 25 × 1 × 1
        return x


class FactChroma(nn.Module):
    """
    Module that uses asymmetric factorization for chroma extraction
    """

    def __init__(self, in_channels: int, channels_12x9: int) -> None:
        super(FactChroma, self).__init__()
        c12x9 = channels_12x9
        self.branch12x9_1 = BasicConv2d(in_channels, c12x9, kernel_size=1)
        self.branch12x9_2 = BasicConv2d(c12x9, c12x9, kernel_size=(1, 9), padding=0)
        self.branch12x9_3 = BasicConv2d(c12x9, 128, kernel_size=(12, 1), padding=0)

    def forward(self, x: Tensor) -> Tensor:
        branch12x9 = self.branch12x9_1(x)
        branch12x9 = self.branch12x9_2(branch12x9)
        branch12x9 = self.branch12x9_3(branch12x9)
        return branch12x9


class FactorizationAuditory(nn.Module):
    """Deep Auditory Model reduced by factorization
    """

    def __init__(self, num_classes: int = 25) -> None:
        super(FactorizationAuditory, self).__init__()

        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.Conv2d_3a_3x3 = BasicConv2d(32, 64, kernel_size=3)
        self.Conv2d_4a_3x3 = BasicConv2d(64, 64, kernel_size=3)
        #self.Conv2d_5a_12x9 = BasicConv2d(64, 128, kernel_size=(12, 9))
        self.Conv2d_5a_12x9 = FactChroma(64, channels_12x9=128)
        self.Conv2d_6a_1x1 = BasicConv2d(128, 25, kernel_size=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(13, 3))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: Tensor) -> Tensor:
        # N x 1 x 105 x 15
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 105 x 15
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 105 x 15
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 105 x 15
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 105 x 15
        x = self.maxpool1(x)
        # N x 32 × 52 × 15
        x = self.Conv2d_3a_3x3(x)
        # N x 64 × 50 × 13
        x = self.Conv2d_4a_3x3(x)
        # N x 64 × 48 × 11
        x = self.maxpool1(x)
        # N x 64 × 24 × 11
        x = self.Conv2d_5a_12x9(x)
        # N x 128 × 13 × 3
        x = self.dropout(x)
        # N x 128 × 13 × 3
        x = self.Conv2d_6a_1x1(x)
        # N x 25 × 13 × 3
        x = self.avg_pool(x)
        # N x 25 × 1 × 1
        return x


class InceptionChroma(nn.Module):
    def __init__(self, num_classes: int = 25) -> None:
        super(DeepAuditory, self).__init__()

        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_3a_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.Conv2d_1b_3x3 = BasicConv2d(32, 64, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(64, 64, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplemented


model = nn.Sequential(OrderedDict([
    # N x 1 x 105 x 15
    ('conv1', nn.Conv2d(1, 32, 3, padding=1)),
    ('relu1', nn.ReLU()),
    ('bnorm1', nn.BatchNorm2d(32)),
    # N x 32 x 105 x 15
    ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
    ('relu2', nn.ReLU()),
    ('bnorm2', nn.BatchNorm2d(32)),
    # N x 32 x 105 x 15
    ('conv3', nn.Conv2d(32, 32, 3, padding=1)),
    ('relu3', nn.ReLU()),
    ('bnorm3', nn.BatchNorm2d(32)),
    # N x 32 x 105 x 15
    ('conv4', nn.Conv2d(32, 32, 3, padding=1)),
    ('relu4', nn.ReLU()),
    ('bnorm4', nn.BatchNorm2d(32)),
    # N x 32 x 105 x 15
    ('pool1', nn.MaxPool2d(kernel_size=(2, 1))),
    ('dropout1', nn.Dropout(0.5)),
    # N x 32 x 52 x 15
    ('conv5', nn.Conv2d(32, 64, 3)),
    ('relu5', nn.ReLU()),
    ('bnorm5', nn.BatchNorm2d(64)),
    # N x 64 x 48 x 11
    ('conv6', nn.Conv2d(64, 64, 3)),
    ('relu6', nn.ReLU()),
    ('bnorm6', nn.BatchNorm2d(64)),
    # N x 64 × 24 × 11
    ('pool2', nn.MaxPool2d(kernel_size=(2, 1))),
    ('dropout2', nn.Dropout(0.5)),
    ('conv7', torch.nn.Conv2d(64, 128, kernel_size=(12, 9))),
    ('relu7', nn.ReLU()),
    ('bnorm7', nn.BatchNorm2d(128)),
    ('dropout3', nn.Dropout(0.5)),
    ('conv8', torch.nn.Conv2d(128, 25, 1)),
    ('pool3', nn.AvgPool2d(kernel_size=(13, 3))),
]))
