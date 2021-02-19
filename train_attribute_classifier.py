import torch
from torch.utils.data import DataLoader

from attribute_game.models import FeatureEncoder
from datasets.AttributeDataset import AttributeDataset


n_attributes = 3
size_attributes = 4
n_epochs = 50
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