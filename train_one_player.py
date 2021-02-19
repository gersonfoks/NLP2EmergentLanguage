import torch
from torch.utils.data import DataLoader

from attribute_game.models import FeatureEncoder, SenderRnn, ReceiverRnn, ReceiverFixed, SenderFixed, OnePlayer
from datasets.AttributeDataset import AttributeDataset, AttributeGameDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_attributes = 4
size_attributes = 3


def train_feature_encoder():
    n_epochs = 5

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
n_symbols = 2
n_receiver = 2

feature_encoder_receiver = train_feature_encoder()

one_player = OnePlayer(feature_encoder_receiver, n_receiver, ).to(device)
dataset = AttributeGameDataset(n_attributes, size_attributes, n_receiver=n_receiver, samples_per_epoch=int(10e3))

train_dataloader = DataLoader(dataset, batch_size=64)

loss_module = torch.nn.CrossEntropyLoss()
parameters = list(one_player.parameters())

optimizer = torch.optim.Adam(
    parameters,
    lr=0.01)

### The actual training loop
n_epochs = 100


print("start training")
# The actual training:
for i in range(n_epochs):
    batch_count = 0
    train_accuracy = 0
    total_loss_batch = 0
    for sender_img, receiver_imgs, target in train_dataloader:
        one_player.zero_grad()

        target = target.to(device)
        sender_img = sender_img.to(device)
        receiver_imgs = [receiver_img.to(device) for receiver_img in receiver_imgs]

        out, out_probs = one_player(sender_img, receiver_imgs)

        loss = loss_module(out, target)
        loss.backward()
        predicted_indices = torch.argmax(out_probs, dim=-1)

        total_loss_batch += loss.item()

        optimizer.step()
        correct = (predicted_indices == target).sum().item()
        train_accuracy += correct / len(target)
        batch_count += 1

    print(train_accuracy / batch_count)
    print(total_loss_batch / batch_count)
    dataset.reset()
