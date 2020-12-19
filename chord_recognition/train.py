import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from livelossplot import PlotLosses


def get_weighted_random_sampler(targets, train_targets):
    _, class_counts = np.unique(targets, return_counts=True)

    weights = 1. / (np.array(class_counts, dtype='float'))
    sampling_weights = [weights[target] for target in train_targets]
    sampler = WeightedRandomSampler(
        weights=sampling_weights,
        num_samples=len(sampling_weights))
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
        self._reset()

    def train(self):
        liveloss = PlotLosses()
        best_acc = 0.0
        for e in range(self.epochs):
            logs = {}
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # put model to training mode
                else:
                    self.model.eval()

                running_loss, running_corrects = self._step(phase)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.float() / len(self.dataloaders[phase].dataset)

                prefix = ''
                if phase == 'val':
                    prefix = 'val_'
                    is_best = epoch_acc > best_acc
                    best_acc = max(epoch_acc, best_acc)
                    self._save_checkpoint(is_best)

                logs[prefix + ' log loss'] = epoch_loss.item()
                logs[prefix + 'accuracy'] = epoch_acc.item()

            liveloss.update(logs)
            liveloss.send()

    def _step(self, phase):
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.dataloaders[phase]:
            inputs = inputs.to(device=self.device, dtype=torch.float32)
            labels = labels.to(device=self.device, dtype=torch.long)
            if phase == 'train':
                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                self.optimizer.zero_grad()

            scores = self.model(inputs)
            scores = scores.squeeze(3).squeeze(2)
            loss = F.cross_entropy(scores, labels)

            _, preds = torch.max(scores, 1)
            running_corrects += torch.sum(preds == labels)

            if phase == 'train':
                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                self.optimizer.step()

            running_loss += loss.detach() * inputs.size(0)

        return running_loss, running_corrects

    def _reset(self):
        pass

    def _save_checkpoint(self, is_best):
        if is_best:
            torch.save(
                self.model.state_dict(),
                f'chord_recognition/models/{self.trained_model_name}'
            )
