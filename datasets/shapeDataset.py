from itertools import permutations, product

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import numpy as np

from gen_shapes_data import COLORS, SHAPES, make_img_one_shape


class ShapeDataset(Dataset):
    '''
    The dataset for a simple mnist signlalling game.
    Each image gets n_receiver-1 other images to be compared with.
    '''

    def __init__(self, samples_per_epoch=10e4, picture_size=32, shape_size=8, transform=None):
        self.samples_per_epoch = int(samples_per_epoch)

        self.picture_size = picture_size
        self.shape_size = shape_size
        self.possible_coordinates = [i * shape_size for i in range(int(picture_size / shape_size))]

        self.possible_items = list(product(COLORS, SHAPES))

        self.items, self.targets = self.generate_items()
        self.transform = transform

    def generate_items(self):
        # First generate the pairs we want

        colors = np.random.choice(COLORS, self.samples_per_epoch)
        shapes = np.random.choice(SHAPES, self.samples_per_epoch)
        x_coordinates = np.random.choice(self.possible_coordinates, self.samples_per_epoch)
        y_coordinates = np.random.choice(self.possible_coordinates, self.samples_per_epoch)

        classes = [
            self.possible_items.index((col, shape)) for col, shape in zip(colors, shapes)
        ]

        items = [
            make_img_one_shape(x, y, col, shape, size=self.shape_size, picture_size=self.picture_size) for
            x, y, col, shape in
            zip(x_coordinates, y_coordinates, colors, shapes)
        ]

        return items, classes

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.transform:
            item = self.transform(item)

        return item, self.targets[idx]


class ShapeGameDataset(Dataset):
    '''
    The dataset for a simple mnist signlalling game.
    Each image gets n_receiver-1 other images to be compared with.
    '''

    def __init__(self, samples_per_epoch=10e4, n_receiver=3, picture_size=32, shape_size=8, transform=None):
        self.samples_per_epoch = int(samples_per_epoch)

        self.n_receiver = n_receiver

        self.picture_size = picture_size
        self.shape_size = shape_size
        self.possible_coordinates = [i * shape_size for i in range(int(picture_size / shape_size))]

        self.sender_items, self.receiver_items, self.targets = self.generate_items()
        self.transform = transform

    def generate_items(self):
        # First generate the pairs we want

        sender_items = []
        receiver_items = []
        targets = []

        possible_items = list(product(COLORS, SHAPES))
        for i in range(self.samples_per_epoch):
            item_ids = np.random.choice(len(possible_items), self.n_receiver, replace=False)
            x_coordinates = np.random.choice(self.possible_coordinates, self.n_receiver)
            y_coordinates = np.random.choice(self.possible_coordinates, self.n_receiver)
            items = [
                make_img_one_shape(x, y, possible_items[id][0], possible_items[id][1], size=self.shape_size,
                                   picture_size=self.picture_size) for
                x, y, id in
                zip(x_coordinates, y_coordinates, item_ids)
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
        receiver_item = self.receiver_items[idx]
        if self.transform:
            sender_item = self.transform(sender_item)
            receiver_item = [
                self.transform(item) for item in receiver_item
            ]

        return sender_item, receiver_item, self.targets[idx]

    def reset(self):
        self.sender_items, self.receiver_items, self.targets = self.generate_items()
