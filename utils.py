
import torch
from torch import nn





def train_hidden_state_model(hidden_state_model, device, train_dataloader, n_epochs):
    '''
    Function to pretrain the hidden state model. 
    :param hidden_state_model: 
    :param device: 
    :param train_dataloader: 
    :param n_epochs: 
    :return: 
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
