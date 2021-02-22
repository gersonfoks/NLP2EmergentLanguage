import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from attribute_game.models import FeatureEncoder, PredictionRNN
from attribute_game.receiver import ReceiverLSTM, ReceiverFixed, ReceiverPredictor
from attribute_game.sender import SenderFixed, SenderRnn
from datasets.AttributeDataset import AttributeDataset

import numpy as np


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

    encoder = get_pretrained_feature_encoder(n_attributes, attributes_size, n_epochs=pretrain_n_epochs,
                                             hidden_state_size=encoder_hidden_state_size)
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

    encoder = get_pretrained_feature_encoder(n_attributes, attributes_size, n_epochs=pretrain_n_epochs,
                                             hidden_state_size=encoder_hidden_state_size)
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

    encoder = get_pretrained_feature_encoder(n_attributes, attributes_size, n_epochs=pretrain_n_epochs,
                                             hidden_state_size=encoder_hidden_state_size)

    return ReceiverPredictor(encoder, n_receiver, n_symbols=n_symbols, msg_len=msg_len, ).to(device)


def pack(msg, msg_len):
    lengths = get_lengths(msg, msg_len)

    msg_packed = pack_padded_sequence(msg, lengths, enforce_sorted=False)

    return msg_packed


def get_lengths(msg, msg_len):
    def find_first(row, ):

        r = np.where(row == 0)
        if len(r[0]) > 0:
            return r[0][0] + 1
        else:
            return msg_len

    ### Get the symbols
    symbol_tensor = torch.argmax(msg, dim=-1)

    symbol_np = symbol_tensor.permute(1, 0).cpu().numpy()
    lengths = []
    for row in symbol_np:
        lengths.append(find_first(row))

    lengths = np.array(lengths)
    return lengths
