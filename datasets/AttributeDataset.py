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
        targets = np.random.choice(self.n_classes, self.samples_per_epoch, replace=True)
        sender_items = [self.to_tensor(self.class_indexes[t]) for t in targets]

        return sender_items, targets

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        sender_item = self.sender_items[idx]

        return sender_item, self.targets[idx]

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

    def __init__(self, n_attributes, size_attributes, n_receiver=3, samples_per_epoch=int(10e4), transform=None, n_remove_classes=0, train=True):
        self.samples_per_epoch = samples_per_epoch
        self.n_receiver = n_receiver
        self.transform = transform
        self.n_attributes = n_attributes
        self.size_attributes = size_attributes
        if n_remove_classes > size_attributes:
            raise AttributeError("number of held out classes should not exceed the number of possible attribute values")
        self.n_classes = (size_attributes ** n_attributes)
        self.n_remove_classes = n_remove_classes

        self.attribute_indexes = [
            [i for i in range(self.size_attributes)] for j in range(self.n_attributes)
        ]

        self.permutations = list(product(*self.attribute_indexes))
        self.class_indexes = {i: att for i, att in enumerate(self.permutations)}

        remove_classes_train = set([tuple([i for j in range(n_attributes)]) for i in range(size_attributes)][:n_remove_classes])
        self.keep_classes = []
        if train:
            for i in range(len(self.permutations)):
                if self.permutations[i] not in remove_classes_train:
                    self.keep_classes.append(i)          
        else:
            for i in range(len(self.permutations)):
                if self.permutations[i] in remove_classes_train:
                    self.keep_classes.append(i)
        

        self.sender_items, self.receiver_items, self.targets = self.generate_items()

    def generate_items(self):
        # First generate the pairs we want
        sender_items = []
        receiver_items = []
        targets = []
        for i in range(self.samples_per_epoch):
            item_ids = np.random.choice(self.keep_classes, self.n_receiver, replace=False) 
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


def get_attribute_game(n_attributes, size_attributes, samples_per_epoch_train=int(10e4),
                       samples_per_epoch_test=int(10e3), batch_size=32):
    '''
    Get a dataloader for the signalling Game
    '''

    signalling_game_train = AttributeGameDataset(n_attributes, size_attributes,
                                                 samples_per_epoch=samples_per_epoch_train, )
    signalling_game_test = AttributeGameDataset(n_attributes, size_attributes, samples_per_epoch=samples_per_epoch_test)

    train_dataloader = DataLoader(signalling_game_train, shuffle=True, batch_size=batch_size, )
    test_dataloader = DataLoader(signalling_game_test, shuffle=False, batch_size=batch_size, )

    return train_dataloader, test_dataloader
