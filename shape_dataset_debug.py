from torchvision.transforms import transforms

from shapeDataset import ShapeDataset, ShapeGameDataset

transform = transforms.Compose([transforms.ToTensor()])
shape_dataset = ShapeDataset(samples_per_epoch=10, picture_size=64, shape_size=16, transform=transform)

transform = transforms.Compose([transforms.ToTensor()])
shape_game = ShapeGameDataset(samples_per_epoch=10, picture_size=64, shape_size=16, )

for (sender_item, receiver_items, target) in shape_game:
    sender_item.show()
    for item in receiver_items:
        item.show()
    print(target)
    break