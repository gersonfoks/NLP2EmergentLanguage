import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.models import HiddenStateModel, VisualModel
from datasets.shapeDataset import ShapeDataset
from utils import train_hidden_state_model

transform = transforms.Compose([transforms.ToTensor()])
data = ShapeDataset(transform=transform)
train_dataloader = DataLoader(data, shuffle=True, batch_size=32, )

hidden_state_model = VisualModel(9, input_channels=3)

n_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_hidden_state_model(hidden_state_model, device, train_dataloader, n_epochs)