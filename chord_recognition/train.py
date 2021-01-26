import os

from livelossplot import PlotLosses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from .ann_utils import ChordModel
from .evaluate import compute_cer, compute_wer
from .utils import ctc_greedy_decoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def get_weighted_random_sampler(labels, labels_train):
    """
    Make a sample to work with unbalanced dataset.
    Each batch should have balanced class distribution.

    Args:
        labels - list of all available labels
        labels_train - labels of a train dataset
    """
    _, class_counts = np.unique(labels, return_counts=True)

    weights = 1. / (np.array(class_counts, dtype='float'))
    train_sampling_weights = [weights[label] for label in labels_train]

    # If `replacement=True`, the sampler oversamples the minority classes with replacement,
    # such that the samples will be repeated.
    # If you don’t want to repeat samples you can specify `replacement=False`
    # and adapt the num_samples to get fewer balanced batches.
    sampler = WeightedRandomSampler(
        weights=train_sampling_weights,
        num_samples=len(train_sampling_weights),
        replacement=True)
    return sampler


def data_processing(data):
    inputs = []
    labels = []
    input_lengths = []
    label_lengths = []
    for batch in data:
        inputs_b, labels_b = batch
        inputs.append(torch.from_numpy(inputs_b))
        labels.append(torch.from_numpy(labels_b))
        input_lengths.append(len(inputs_b) // 2)
        label_lengths.append(len(labels_b))
    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return inputs, labels, input_lengths, label_lengths


class Solver:
    """
    Encapsulates all the logic necessary for a training.
    """

    def __init__(
        self,
        model,
        optimizer,
        dataloaders,
        scheduler=None,
        loss=None,
        epochs=1,
        trained_model_name='best_model.pth',
    ):
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(device=self.device)
        self.trained_model_name = trained_model_name
        self.loss_name = loss
        self.criterion = nn.CTCLoss()
        self.chord_model = ChordModel()
        self._reset()

    def train(self):
        liveloss = PlotLosses()
        best_loss = np.inf
        for e in range(self.epochs):
            logs = {}
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # put model to training mode
                else:
                    self.model.eval()

                running_loss, avg_wer = self._step(phase)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)

                prefix = ''
                if phase == 'val':
                    prefix = 'val_'
                    is_best = epoch_loss < best_loss
                    best_loss = epoch_loss
                    self._save_checkpoint(is_best)

                logs[prefix + ' log loss'] = epoch_loss.item()
                logs[prefix + ' WER'] = avg_wer

            liveloss.update(logs)
            liveloss.send()

    def _step(self, phase):
        running_loss = 0.0
        running_wer = []

        for data in self.dataloaders[phase]:
            inputs, labels, input_lengths, label_lengths = data
            inputs = inputs.to(device=self.device, dtype=torch.float32)
            labels = labels.to(device=self.device, dtype=torch.long)
            if phase == 'train':
                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                self.optimizer.zero_grad()

            out = self.model(inputs)
            out = F.log_softmax(out, dim=2)

            loss = self.criterion(out, labels, input_lengths, label_lengths)

            with torch.no_grad():
                decoded_out = ctc_greedy_decoder(out.cpu().numpy())
                labels = labels.cpu().numpy()
                _, N, _ = out.shape
                for n in range(N):
                    target_labels = self.chord_model.onehot_to_labels(labels[n][:label_lengths[n]])
                    out_labels = self.chord_model.onehot_to_labels(decoded_out[n])
                    wer = compute_wer(target_labels, out_labels)
                    running_wer.append(wer)

            if phase == 'train':
                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

            running_loss += loss.detach() * inputs.size(0)

        avg_wer = sum(running_wer) / len(running_wer)
        return running_loss, avg_wer

    def _reset(self):
        pass

    def _save_checkpoint(self, is_best):
        if is_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(MODELS_DIR, self.trained_model_name)
            )
