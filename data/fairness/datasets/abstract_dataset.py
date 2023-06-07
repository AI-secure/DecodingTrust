import random
from abc import ABC, abstractmethod
from os import makedirs, path

import numpy as np
import torch
import torch.utils.data

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

project_root = path.dirname(path.dirname(path.abspath(__file__)))


class AbstractDataset(ABC, torch.utils.data.Dataset):

    @abstractmethod
    def __init__(self, name, split):
        if split not in ['train', 'test', 'validation']:
            raise ValueError('Unknown dataset split')

        self.split = split
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = path.join(project_root, 'data', name)
        makedirs(self.data_dir, exist_ok=True)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.protected[index]

    def __len__(self):
        return self.labels.size()[0]

    def _normalize(self, columns):
        columns = columns if columns is not None else np.arange(self.X_train.shape[1])

        self.mean, self.std = self.X_train.mean(dim=0)[columns], self.X_train.std(dim=0)[columns]

        self.X_train[:, columns] = (self.X_train[:, columns] - self.mean) / self.std
        self.X_val[:, columns] = (self.X_val[:, columns] - self.mean) / self.std
        self.X_test[:, columns] = (self.X_test[:, columns] - self.mean) / self.std

    def _assign_split(self):
        if self.split == 'train':
            self.features, self.labels, self.protected = self.X_train, self.y_train, self.protected_train
        elif self.split == 'test':
            self.features, self.labels, self.protected = self.X_test, self.y_test, self.protected_test
        elif self.split == 'validation':
            self.features, self.labels, self.protected = self.X_val, self.y_val, self.protected_val

        self.features = self.features.float()
        self.labels = self.labels.float()
        self.protected = self.protected.long()

    def pos_weight(self, split):
        if split == 'train':
            labels = self.y_train
        elif split == 'train-val':
            labels = torch.cat((self.y_train, self.y_val))
        else:
            raise ValueError('Unknown split')

        positives = torch.sum(labels == 1).float()
        negatives = torch.sum(labels == 0).float()

        assert positives + negatives == labels.shape[0]

        return negatives / positives
