import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from models.models import SenderModelFixedLength, ReceiverModuleFixedLength, PredictionRNN, SenderRnn, HiddenStateModel
from signalling_game import SignallingGameDataset


def train_hidden_state_model(hidden_state_model, device, train_dataloader, n_epochs):
    '''
    Function to pretrain the hidden state model. 
    :param hidden_state_model: model to train 
    :param device: on which device the tensors shoudl go
    :param train_dataloader: training dataloader
    :param n_epochs: number of epochs the model should train
    :return: None
    '''''

    hidden_state_model = hidden_state_model.to(device)

    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hidden_state_model.parameters(), lr=0.001)

    for n in range(n_epochs):
        train_accuracy = 0
        batch_count = 0
        for inputs, targets in train_dataloader:
            # Prepare data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Calculate model output and loss
            predictions, prediction_probabilities = hidden_state_model.to_predictions(inputs)

            loss = loss_module(predictions, targets)

            accuracyPredictions = torch.argmax(prediction_probabilities, dim=-1)
            correct = (accuracyPredictions == targets).sum().item()
            train_accuracy += correct / len(targets)

            # Execute backwards propagation
            hidden_state_model.zero_grad()
            loss.backward()
            optimizer.step()

            batch_count += 1

        print(train_accuracy / batch_count)


def get_mnist_signalling_game(batch_size=32, size=None):
    '''
    Get a dataloader for the signalling Game
    '''
    transform = transforms.Compose([transforms.ToTensor()])
    signalling_game_train = SignallingGameDataset(transform=transform)
    signalling_game_test = SignallingGameDataset(train=False, transform=transform)

    if size:
        indices = [i for i in range(size)]
        signalling_game_train = Subset(signalling_game_train, indices)
        signalling_game_test = Subset(signalling_game_test, indices)
    train_dataloader = DataLoader(signalling_game_train, shuffle=True, batch_size=batch_size, )
    test_dataloader = DataLoader(signalling_game_test, shuffle=False, batch_size=batch_size, )

    return train_dataloader, test_dataloader


def get_sender(n_symbols, msg_len, device, fixed_size=True, pretrain=None):
    '''
    Get the sender model
    '''
    sender = None

    hidden_state_model = None
    if pretrain:
        if pretrain == 'MNIST':
            hidden_state_model = get_mnist_pretrain(device)

    if fixed_size:
        sender = SenderModelFixedLength(10, n_symbols=n_symbols, msg_len=msg_len, hidden_state_model=hidden_state_model).to(device)
    else:
        sender = SenderRnn(10, n_symbols=n_symbols, msg_len=msg_len)
    return sender


def get_receiver(n_symbols, msg_len, device, pretrain=True):
    '''
    Get the receiver model
    '''
    hidden_state_model = None
    if pretrain:
        if pretrain == 'MNIST':
            hidden_state_model = get_mnist_pretrain(device)

    receiver = ReceiverModuleFixedLength(10, n_symbols=n_symbols, msg_len=msg_len, hidden_state_model=hidden_state_model).to(device)
    return receiver


def get_predictor(n_symbols, hidden_size, device):
    '''
    Get the predictor model
    '''
    predictor = PredictionRNN(n_symbols, hidden_size).to(device)
    return predictor


def get_mnist_pretrain(device, n_epochs=2, root='./data/'):
    transform = transforms.Compose([transforms.ToTensor()])
    data = MNIST(root=root, download=True, train=True, transform=transform)
    train_dataloader = DataLoader(data, shuffle=True, batch_size=32, )

    hidden_state_model = HiddenStateModel(10)

    train_hidden_state_model(hidden_state_model, device, train_dataloader, n_epochs)
    return hidden_state_model