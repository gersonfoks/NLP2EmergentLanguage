import torch
from torch.utils.data import DataLoader

from datasets.MsgDataset import MarkovProcess, MsgDataset
from shape_game.models.models import PredictionRNN

simple_transitions = {
    0: {1: 0.5, 2: 0.5},
    1: {3: 1.0},
    2: {3: 1.0}

}
stop_symbol = 3

mp = MarkovProcess(simple_transitions, stop_symbol, )

msg_dataset = MsgDataset(mp, )

train_dataloader = DataLoader(msg_dataset, shuffle=True, batch_size=32, )

### Now we train the rnn.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rnn = PredictionRNN(4, 18).to(device)
loss_module = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

n_epochs = 100

for n in range(n_epochs):
    train_accuracy = 0
    batch_count = 0
    for msgs in train_dataloader:
        # Prepare data
        msgs = msgs.to(device)

        targets = msgs[:, 1:]
        msgs = torch.nn.functional.one_hot(msgs).float()
        msgs_input = msgs[:, :-1]

        # Calculate model output and loss
        predictions, prediction_probabilities, hidden = rnn(msgs_input)

        prediction_probabilities = prediction_probabilities.reshape(predictions.shape[0] * (predictions.shape[1]), -1)
        predictions = predictions.reshape(predictions.shape[0] * (predictions.shape[1]), -1)

        targets = targets.reshape(-1)
        loss = loss_module(predictions, targets)

        accuracyPredictions = torch.argmax(prediction_probabilities, dim=-1)
        correct = (accuracyPredictions == targets).sum().item()
        train_accuracy += correct / len(targets)

        # Execute backwards propagation
        rnn.zero_grad()
        loss.backward()
        optimizer.step()

        batch_count += 1

    print(train_accuracy / batch_count)
