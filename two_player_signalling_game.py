from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


import torch
from torch import nn
import numpy as np

from models.models import HiddenStateModel, SenderModelFixedLength, ReceiverModuleFixedLength
from signalling_game import SignallingGameDataset

# Seed everything for reproducibility
from utils import train_hidden_state_model

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

batch_size = 128

n_epochs = 5
root = './data/'

download = True
train = True

transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST(root=root, download=download, train=train, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First we train the sender_model on mnist to get some ground knowledge
train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, )




### We first train the hidden state model, which we use in both the sender and the receiver.
hidden_state_model_1 = HiddenStateModel(10).to(device)
hidden_state_model_2 = HiddenStateModel(10).to(device)

pretrain = False

print("start training hidden states")
if pretrain:
    train_hidden_state_model(hidden_state_model_1, device, train_dataloader, n_epochs)
    train_hidden_state_model(hidden_state_model_2, device, train_dataloader, n_epochs)




# hidden_state_model_1.eval()
# hidden_state_model_2.eval()

## Now the real fun begins. Lets play the real game
signalling_game = SignallingGameDataset(transform=transform)


signalling_game_dataloader = DataLoader(signalling_game, batch_size=128)

sender = SenderModelFixedLength(10, hidden_state_model=hidden_state_model_1).to(device)

receiver = ReceiverModuleFixedLength(10, hidden_state_model=hidden_state_model_2).to(device)

loss_module = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(sender.parameters()) + list(receiver.parameters()), lr=0.0001)

n_epochs = 10
print("start training sender and receiver")
for n in range(n_epochs):
    train_accuracy = 0
    batch_count = 0
    total_loss = 0
    for x,xs, target in signalling_game_dataloader:
        x = x.to(device)
        xs = [t.to(device) for t in xs]
        targets = target.to(device)
        msg = sender(x)



        out, out_probs = receiver(xs, msg)
        loss = loss_module(out, targets)

        total_loss += loss.item()

        accuracyPredictions = torch.argmax(out_probs, dim=-1)


        correct = (accuracyPredictions == targets).sum().item()
        train_accuracy += correct / len(targets)

        # Execute backwards propagation
        sender.zero_grad()
        receiver.zero_grad()
        loss.backward()
        optimizer.step()

        batch_count += 1

    print("train_acc")
    print(train_accuracy / batch_count)
    print("loss")
    print(total_loss / batch_count)
