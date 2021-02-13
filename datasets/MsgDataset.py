import torch
from torch.utils.data import Dataset
import numpy as np


class MsgDataset(Dataset):
    '''
    The dataset for a simple mnist signlalling game.
    Each image gets n_receiver-1 other images to be compared with.
    '''

    def __init__(self, generating_process, samples_per_epoch=100, transform=None, ):
        self.samples_per_epoch = int(samples_per_epoch)

        self.generating_process = generating_process

        self.items = self.generate_items()
        self.transform = transform

    def generate_items(self):
        return torch.tensor([self.generating_process.get_new_sequence() for i in range(self.samples_per_epoch)])

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.transform:
            item = self.transform(item)

        return item


class MarkovProcess:

    def __init__(self, transitions, end_symbol):
        self.transitions = transitions
        self.end_symbol = end_symbol

    def get_new_sequence(self):
        sequence = [0]
        current_symbol = 0  ## The start symbol
        while current_symbol != self.end_symbol:
            next_states = list(self.transitions[current_symbol].keys())
            probabilities = [self.transitions[current_symbol][s] for s in next_states]
            current_symbol = np.random.choice(next_states, p=probabilities)
            sequence.append(current_symbol)
        return sequence
