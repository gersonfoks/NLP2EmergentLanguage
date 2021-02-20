import torch
from torch.utils.data import DataLoader

from attribute_game.models import SenderFixed, SenderRnn, FeatureEncoder, ReceiverFixed, PredictionRNN, \
    ReceiverPredictor
from attribute_game.rnn_receiver import ReceiverLSTM
from datasets.AttributeDataset import AttributeDataset


def get_pretrained_feature_encoder(n_attributes, size_attributes, n_epochs=3, hidden_state_size=128):
    dataset = AttributeDataset(n_attributes, size_attributes, samples_per_epoch=1000)

    train_dataloader = DataLoader(dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = FeatureEncoder(n_attributes, size_attributes, hidden_state_size=hidden_state_size).to(device)

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
        print("accuracy")
        print(train_accuracy / batch_count)
    return classifier


def get_sender(n_attributes, attributes_size, n_symbols, msg_len, device, fixed_size=True, pretrain_n_epochs=3,
               encoder_hidden_state_size=128):
    '''
    Get the sender model
    '''

    encoder = get_pretrained_feature_encoder(n_attributes, attributes_size, n_epochs=pretrain_n_epochs, hidden_state_size=encoder_hidden_state_size )
    if fixed_size:
        sender = SenderFixed(encoder, n_symbols=n_symbols, msg_len=msg_len,
                             ).to(device)
    else:
        sender = SenderRnn(encoder, n_symbols=n_symbols, msg_len=msg_len, ).to(device)
    return sender


def get_receiver(n_attributes, attributes_size, n_receiver, n_symbols, msg_len, device, fixed_size=True,
                 pretrain_n_epochs=3, encoder_hidden_state_size=128):
    '''
    Get the sender model
    '''

    encoder = get_pretrained_feature_encoder(n_attributes, attributes_size,  n_epochs=pretrain_n_epochs, hidden_state_size=encoder_hidden_state_size)
    if fixed_size:
        receiver = ReceiverFixed(encoder, n_receiver, n_symbols=n_symbols, msg_len=msg_len,
                                 ).to(device)
    else:
        receiver = ReceiverLSTM(encoder, n_receiver, n_symbols=n_symbols, msg_len=msg_len, ).to(device)
    return receiver


def get_predictor(n_symbols, hidden_size, device):
    return PredictionRNN(n_symbols, hidden_size).to(device)


def get_receiver_predictor(n_attributes, attributes_size, n_receiver, n_symbols, msg_len, device, fixed_size=True,
                 pretrain_n_epochs=3, encoder_hidden_state_size=128):
    '''
    Get the receiver predictor
    '''

    encoder = get_pretrained_feature_encoder(n_attributes, attributes_size,  n_epochs=pretrain_n_epochs, hidden_state_size=encoder_hidden_state_size)

    return ReceiverPredictor(encoder, n_receiver, n_symbols=n_symbols, msg_len=msg_len, ).to(device)
