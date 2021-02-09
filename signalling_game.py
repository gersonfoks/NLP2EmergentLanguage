from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import numpy as np


class SignallingGameDataset(Dataset):
    '''
    The dataset for a simple mnist signlalling game.
    Each image gets n_receiver-1 other images to be compared with.
    '''

    def __init__(self, n_receiver=3, train=True, transform=None, root='./data'):
        self.data = MNIST(root=root, download=True, train=train, transform=transform)
        self.n_receiver = n_receiver

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sender_img = self.data[idx][0]

        receiver_indices = np.random.choice(len(self), self.n_receiver - 1)

        receiver_choices = [sender_img] + [self.data[i][0] for i in receiver_indices]
        shuffle = [i for i in range(self.n_receiver)]
        np.random.shuffle(shuffle)

        target = shuffle.index(0)

        receiver_choices = [receiver_choices[i] for i in shuffle]

        return sender_img, receiver_choices, target


