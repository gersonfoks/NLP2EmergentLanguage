from torchvision.transforms import transforms

from shapeDataset import ShapeDataset

transform = transforms.Compose([transforms.ToTensor()])
shape_dataset = ShapeDataset(epoch_size=10, picture_size=64, shape_size=16, transform=transform)

