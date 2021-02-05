from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import torch
from torch import nn
import numpy as np

from models.models import HiddenStateModel, SenderModelFixedLength, ReceiverModuleFixedLength, PredictionRNN
from signalling_game import SignallingGameDataset

# Seed everything for reproducibility
from utils import train_hidden_state_model

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

batch_size = 128

n_epochs = 1
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

pretrain = True

print("start training hidden states")
if pretrain:
    train_hidden_state_model(hidden_state_model_1, device, train_dataloader, n_epochs)
    train_hidden_state_model(hidden_state_model_2, device, train_dataloader, n_epochs)

hidden_state_model_1.eval()
hidden_state_model_2.eval()

## Now the real fun begins. Lets play the real game
signalling_game = SignallingGameDataset(transform=transform)

signalling_game_dataloader = DataLoader(signalling_game, batch_size=128)

sender = SenderModelFixedLength(10, hidden_state_model=hidden_state_model_1).to(device)

receiver = ReceiverModuleFixedLength(10, hidden_state_model=hidden_state_model_2).to(device)


n_words = 3
hidden_size = 32
predictor = PredictionRNN(n_words, hidden_size).to(device)

loss_module = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(sender.parameters()) + list(receiver.parameters()) + list(predictor.parameters()), lr=0.0001)

loss_module_predictor = nn.MSELoss()




n_epochs = 10
print("start training sender and receiver")
for n in range(n_epochs):
    train_accuracy = 0
    batch_count = 0
    total_loss = 0
    total_loss_predictor = 0
    for x, xs, target in signalling_game_dataloader:
        x = x.to(device)
        xs = [t.to(device) for t in xs]
        targets = target.to(device)
        msg = sender(x)

        hidden = predictor.initHidden().to(device)
        predictions, out_probs, hidden = predictor(msg, hidden)


        prediction_squeazed = out_probs.reshape(-1, n_words)

        ### Not sure if this works though, because of the non differentiability
        ### THink we need to make custom loss function.
        #msg_squezed = torch.argmax(msg.reshape(-1, n_words), dim=-1)
        msg_squezed = msg.reshape(-1, n_words)

        loss_predictor = loss_module_predictor(prediction_squeazed, msg_squezed)

        total_loss_predictor += loss_predictor.item()

        out, out_probs = receiver(xs, msg)
        loss = loss_module(out, targets) + 0.1 * loss_predictor

        total_loss += loss.item()

        accuracyPredictions = torch.argmax(out_probs, dim=-1)

        correct = (accuracyPredictions == targets).sum().item()
        train_accuracy += correct / len(targets)

        # Execute backwards propagation
        sender.zero_grad()
        receiver.zero_grad()
        predictor.zero_grad()

        loss.backward()
        optimizer.step()

        batch_count += 1

    print("train_acc")
    print(train_accuracy / batch_count)
    print("loss")
    print(total_loss / batch_count)

    print("loss predictor")
    print(total_loss_predictor / batch_count)