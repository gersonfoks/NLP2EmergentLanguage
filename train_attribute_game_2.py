import torch
from torch.utils.data import DataLoader

from attribute_game.models import FeatureEncoder, SenderRnn, ReceiverFixed, SenderFixed
from attribute_game.rnn_receiver import ReceiverRnn, ReceiverLSTM
from datasets.AttributeDataset import AttributeDataset, AttributeGameDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_attributes = 2
size_attributes = 2


def train_feature_encoder():
    n_epochs = 3

    dataset = AttributeDataset(n_attributes, size_attributes, samples_per_epoch=1000)

    train_dataloader = DataLoader(dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = FeatureEncoder(n_attributes, size_attributes).to(device)

    loss_module = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    for n in range(n_epochs):
        train_accuracy = 0
        batch_count = 0
        for inputs, targets in train_dataloader:
            # Prepare data
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            # Calculate model output and loss
            predictions, prediction_probabilities = classifier.to_predictions(inputs)

            loss = loss_module(predictions, targets)

            accuracyPredictions = torch.argmax(prediction_probabilities, dim=-1)
            correct = (accuracyPredictions == targets).sum().item()
            train_accuracy += correct / len(targets)

            # Execute backwards propagation
            classifier.zero_grad()
            loss.backward()
            optimizer.step()

            batch_count += 1

        print(train_accuracy / batch_count)
    return classifier


msg_len = 2
n_symbols = 250
n_receiver = 2
feature_encoder_sender = train_feature_encoder()
feature_encoder_receiver = train_feature_encoder()

sender =   SenderRnn(feature_encoder_sender, msg_len=msg_len, n_symbols=n_symbols).to(device)
receiver = ReceiverLSTM(feature_encoder_receiver, n_receiver, n_symbols=n_symbols, msg_len=msg_len).to(device)

dataset = AttributeGameDataset(n_attributes, size_attributes, n_receiver=n_receiver, samples_per_epoch=int(10e3))

train_dataloader = DataLoader(dataset, batch_size=8)

loss_module = torch.nn.CrossEntropyLoss()
parameters = list(sender.parameters()) + list(receiver.parameters())

optimizer = torch.optim.Adam(
    parameters,
    lr=0.01)

### The actual training loop
n_epochs = 50

def train(sender, receiver, train_dataloader, loss_module, optimizer, n_epochs):

    # The actual training:
    for i in range(n_epochs):
        batch_count = 0
        train_accuracy = 0
        total_loss_batch = 0
        for sender_img, receiver_imgs, target in train_dataloader:


            sender_img = sender_img.to(device)
            receiver_imgs = [receiver_img.to(device) for receiver_img in receiver_imgs]
            target = target.to(device)
            msg = sender(sender_img)

            out, out_probs = receiver(receiver_imgs, msg)

            loss = loss_module(out, target)
            receiver.zero_grad()
            sender.zero_grad()
            loss.backward()
            optimizer.step()

            predicted_indices = torch.argmax(out_probs, dim=-1)

            total_loss_batch += loss.item()

            correct = (predicted_indices == target).sum().item()
            train_accuracy += correct / len(target)
            batch_count += 1

        print(train_accuracy / batch_count)
        print(total_loss_batch / batch_count)
        dataset.reset()

train(sender, receiver, train_dataloader, loss_module, optimizer, n_epochs)


# print("start training rnn")
#
# feature_encoder_receiver = train_feature_encoder()
# receiver = ReceiverRnn(feature_encoder_receiver, n_receiver, n_symbols=n_symbols, msg_len=msg_len).to(device)
#
# dataset = AttributeGameDataset(n_attributes, size_attributes, n_receiver=n_receiver, samples_per_epoch=int(10e3))
#
# train_dataloader = DataLoader(dataset, batch_size=256)
#
# loss_module = torch.nn.CrossEntropyLoss()
# parameters = list(receiver.parameters()) + list(sender.parameters())
#
# optimizer = torch.optim.Adam(
#     parameters,
#     lr=0.001)
#
# ### The actual training loop
# n_epochs = 20
#
# train(sender, receiver, train_dataloader, loss_module, optimizer, n_epochs)
