from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 32, 3, padding=1)),
    ('relu1', nn.ReLU()),
    ('bnorm1', nn.BatchNorm2d(32)),
    ('conv2', nn.Conv2d(32, 32, 3, padding=1)),
    ('relu2', nn.ReLU()),
    ('bnorm2', nn.BatchNorm2d(32)),
    ('conv3', nn.Conv2d(32, 32, 3, padding=1)),
    ('relu3', nn.ReLU()),
    ('bnorm3', nn.BatchNorm2d(32)),
    ('conv4', nn.Conv2d(32, 32, 3, padding=1)),
    ('relu4', nn.ReLU()),
    ('bnorm4', nn.BatchNorm2d(32)),
    ('pool1', nn.MaxPool2d(kernel_size=(2, 1))),
    ('dropout1', nn.Dropout(0.5)),
    ('conv5', nn.Conv2d(32, 64, 3)),
    ('relu5', nn.ReLU()),
    ('bnorm5', nn.BatchNorm2d(64)),
    ('conv6', nn.Conv2d(64, 64, 3)),
    ('relu6', nn.ReLU()),
    ('bnorm6', nn.BatchNorm2d(64)),
    ('pool2', nn.MaxPool2d(kernel_size=(2, 1))),
    ('dropout2', nn.Dropout(0.5)),
    ('conv7', torch.nn.Conv2d(64, 128, kernel_size=(12, 9))),
    ('relu7', nn.ReLU()),
    ('bnorm7', nn.BatchNorm2d(128)),
    ('dropout3', nn.Dropout(0.5)),
    ('conv8', torch.nn.Conv2d(128, 25, 1)),
    ('pool3', nn.AvgPool2d(kernel_size=(13, 3))),
]))


class MirexNet(nn.Module):
    def __init__(self):
        super(MirexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv5 = torch.nn.Conv2d(32, 64, 3)
        self.conv6 = torch.nn.Conv2d(64, 64, 3)
        self.conv7 = torch.nn.Conv2d(64, 128, kernel_size=(12, 9))
        self.conv8 = torch.nn.Conv2d(128, 25, 1)
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AvgPool2d(kernel_size=(13, 3))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        x = self.avg_pool(x)
        return x
