import os

from livelossplot import PlotLosses
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score

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


class Solver:
    """
    Encapsulates all the logic necessary for a training.
    """

    def __init__(
        self,
        model,
        optimizer,
        learning_rate,
        dataloaders,
        loss=None,
        epochs=1,
        trained_model_name='best_model.pth',
    ):
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(device=self.device)
        self.trained_model_name = trained_model_name
        self.loss_name = loss
        self.criterion = self._get_criterion(loss)
        self._reset()

    def train(self):
        liveloss = PlotLosses()
        best_f1 = 0.0
        best_loss = np.inf
        for e in range(self.epochs):
            logs = {}
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # put model to training mode
                else:
                    self.model.eval()

                running_loss, targets, outputs = self._step(phase)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                #f1 = f1_score(outputs, targets, average='weighted')

                prefix = ''
                if phase == 'val':
                    prefix = 'val_'
                    #is_best = f1 > best_f1
                    #best_f1 = max(f1, best_f1)
                    is_best = epoch_loss < best_loss
                    best_loss = epoch_loss
                    self._save_checkpoint(is_best)

                logs[prefix + ' log loss'] = epoch_loss.item()
                #logs[prefix + ' F1'] = f1

            liveloss.update(logs)
            liveloss.send()

    def _step(self, phase):
        running_loss = 0.0
        targets_list = []
        outputs = []

        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(device=self.device, dtype=torch.float32)
            labels = labels.to(device=self.device, dtype=torch.long)
            if phase == 'train':
                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                self.optimizer.zero_grad()

            out = self.model(inputs)

            if self.loss_name == 'ctc':
                T, N, C = out.shape
                input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
                labels_lengths = torch.full((N,), 8, dtype=torch.long, device=self.device)
                loss = self.criterion(out, labels, input_lengths, labels_lengths)
            elif self.loss_name == 'cross-entropy':
                loss = self.criterion(out, labels)
                preds = torch.argmax(out, 1)

                outputs.append(preds.cpu().data.numpy())
                targets_list.append(labels.cpu().data.numpy())
            else:
                raise ValueError("Invalid loss name")

            if phase == 'train':
                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                self.optimizer.step()

            running_loss += loss.detach() * inputs.size(0)

        if outputs:
            outputs = np.concatenate(outputs)
        if targets_list:
            targets_list = np.concatenate(targets_list)
        return running_loss, targets_list, outputs

    def _reset(self):
        pass

    @staticmethod
    def _encode_inputs(preds):
        """
        Encode inputs for passing to CTC loss
        """
        return preds

    @staticmethod
    def _get_criterion(loss_name):
        losses = {
            'cross-entropy': nn.CrossEntropyLoss(),
            'ctc': nn.CTCLoss(),
        }
        return losses.get(loss_name, 'cross-entropy')

    def _save_checkpoint(self, is_best):
        if is_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(MODELS_DIR, self.trained_model_name)
            )
