import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class FeatureEncoder(nn.Module):

    def __init__(self, n_attributes, size_attributes, hidden_state_size=128):
        super(FeatureEncoder, self).__init__()

        self.n_attributes = n_attributes
        self.size_attributes = size_attributes
        self.n_classes = size_attributes ** n_attributes
        self.hidden_state_size = hidden_state_size

        self.to_hidden = nn.Sequential(
            nn.Linear(self.n_attributes * self.size_attributes, hidden_state_size),
        )
        self.to_prediction = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_state_size, self.n_classes)
        )

    def to_predictions(self, x):
        '''
        Calculates the output of the model given the input
        :param x: input for the model
        :return: torch.tensor: self.all_modules(x)
        '''
        hidden = self.to_hidden(x)
        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs

    def forward(self, x):
        return self.to_hidden(x)





class PredictionRNN(nn.Module):
    def __init__(self, n_words, hidden_size):
        '''
        A RNN that predicts the next symbol in the language
        :param n_words:
        :param hidden_size:
        '''
        super(PredictionRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(n_words, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)

        self.predictions = nn.Linear(hidden_size, n_words)
        self.n_words = n_words

    def forward(self, input):
        batch_size = input.shape[1]
        input = input.view(-1, self.n_words)
        embedded = self.embedding(input)

        embedded = embedded.reshape(-1, batch_size, self.hidden_size)

        out, (hidden, cell_state) = self.rnn(embedded)

        # Each hidden state put trough something to a small nn.
        predictions_logits = self.predictions(out.squeeze(dim=0))

        out_probs = torch.softmax(predictions_logits, dim=-1)

        return predictions_logits, out_probs, hidden.squeeze(dim=0)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class OnePlayer(nn.Module):
    def __init__(self, feature_encoder, n_xs, ):
        '''
        A sender that send fixed length messages
        '''
        super(OnePlayer, self).__init__()
        self.feature_encoder = feature_encoder

        self.n_xs = n_xs

        self.hidden_state_size = self.feature_encoder.hidden_state_size

        self.to_prediction = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_state_size * (self.n_xs + 1), self.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, self.n_xs)
        )

    def forward(self, x, xs):
        hidden_states = [self.feature_encoder(x)]
        for x in xs:
            hidden_state = self.feature_encoder(x)

            hidden_states.append(hidden_state)

        # Permute the msg to make sure that the batch is second

        hidden = torch.cat(hidden_states, dim=1)

        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs
