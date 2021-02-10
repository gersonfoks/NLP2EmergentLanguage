from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import numpy as np

from gen_shapes_data import COLORS, SHAPES, make_img_one_shape


class ShapeDataset(Dataset):
    '''
    The dataset for a simple mnist signlalling game.
    Each image gets n_receiver-1 other images to be compared with.
    '''

    def __init__(self, epoch_size=10e4, picture_size=32, shape_size=8, transform=None):
        self.epoch_size = epoch_size

        self.picture_size = picture_size
        self.shape_size = shape_size
        self.possible_coordinates = [i * shape_size for i in range(int(picture_size / shape_size))]

        self.items, self.targets = self.generate_items()
        self.transform = transform

    def generate_items(self):
        # First generate the pairs we want

        colors = np.random.choice(COLORS, self.epoch_size)
        shapes = np.random.choice(SHAPES, self.epoch_size)
        x_coordinates = np.random.choice(self.possible_coordinates, self.epoch_size)
        y_coordinates = np.random.choice(self.possible_coordinates, self.epoch_size)

        classes = [
            (col, shape) for col, shape in zip(colors, shapes)
        ]

        items = [
            make_img_one_shape(x, y, col, shape, size=self.shape_size, picture_size=self.picture_size) for
            x, y, col, shape in
            zip(x_coordinates, y_coordinates, colors, shapes)
        ]

        return items, classes

    def __len__(self):
        return len(self.epoch_size)

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.transform:
            item = self.transform(item)

        return item, self.targets[idx]
