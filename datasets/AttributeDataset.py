from itertools import permutations, product

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import numpy as np


class AttributeDataset(Dataset):
    '''
    The dataset for a simple attribute passing game.
    '''

    def __init__(self, n_attributes, size_attributes, n_receiver=3, samples_per_epoch=int(10e4), transform=None):
        self.samples_per_epoch = samples_per_epoch
        self.n_receiver = n_receiver
        self.transform = transform
        self.n_attributes = n_attributes
        self.size_attributes = size_attributes
        self.n_classes = size_attributes ** n_attributes

        self.attribute_indexes = [
            [i for i in range(self.size_attributes)] for j in range(self.n_attributes)
        ]

        self.permutations = list(product(*self.attribute_indexes))
        self.class_indexes = {i: att for i, att in enumerate(self.permutations)}

        self.sender_items, self.targets = self.generate_items()

    def generate_items(self):
        # First generate the pairs we want
        sender_items = []
        receiver_items = []
        targets =  np.random.choice(self.n_classes, self.samples_per_epoch, replace=True)
        sender_items = [self.to_tensor(self.class_indexes[t]) for t in targets]

        return sender_items, targets

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        sender_item = self.sender_items[idx]

        return sender_item,  self.targets[idx]

    def to_tensor(self, attributes):
        attribute_tensor = torch.zeros(self.n_attributes * self.size_attributes)
        for i, att in enumerate(attributes):
            attribute_tensor[i * self.size_attributes + att] = 1
        return attribute_tensor

    def reset(self):
        self.sender_items, self.receiver_items, self.targets = self.generate_items()


class AttributeGameDataset(Dataset):
    '''
    The dataset for a simple attribute passing game.
    '''

    def __init__(self, n_attributes, size_attributes, n_receiver=3, samples_per_epoch=int(10e4), transform=None):
        self.samples_per_epoch = samples_per_epoch
        self.n_receiver = n_receiver
        self.transform = transform
        self.n_attributes = n_attributes
        self.size_attributes = size_attributes
        self.n_classes = size_attributes ** n_attributes

        self.attribute_indexes = [
            [i for i in range(self.size_attributes)] for j in range(self.n_attributes)
        ]

        self.permutations = list(product(*self.attribute_indexes))
        self.class_indexes = {i: att for i, att in enumerate(self.permutations)}

        self.sender_items, self.receiver_items, self.targets = self.generate_items()

    def generate_items(self):
        # First generate the pairs we want
        sender_items = []
        receiver_items = []
        targets = []
        for i in range(self.samples_per_epoch):
            item_ids = np.random.choice(self.n_classes, self.n_receiver, replace=False)
            items = [
                self.to_tensor(self.class_indexes[id]) for id in item_ids
            ]
            target_index = int(np.random.choice(self.n_receiver, 1)[0])
            sender_items.append(items[target_index])
            receiver_items.append(items)
            targets.append(target_index)

        return sender_items, receiver_items, targets

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        sender_item = self.sender_items[idx]

        return sender_item, self.receiver_items[idx], self.targets[idx]

    def to_tensor(self, attributes):
        attribute_tensor = torch.zeros(self.n_attributes * self.size_attributes)
        for i, att in enumerate(attributes):
            attribute_tensor[i * self.size_attributes + att] = 1
        return attribute_tensor

    def reset(self):
        self.sender_items, self.receiver_items, self.targets = self.generate_items()
