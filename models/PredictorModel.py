

import torch
from torch import nn


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
        self.gru = nn.LSTM(hidden_size, hidden_size)

        self.predictions = nn.Linear(hidden_size, n_words)
        self.n_words = n_words

    def forward(self, input):
        batch_size = input.shape[0]

        input = input.reshape(-1, self.n_words)

        embedded = self.embedding(input)

        embedded = embedded.reshape(-1, batch_size, self.hidden_size)

        out, (hidden, cell_state) = self.gru(embedded)

        out = out[:-1]

        # Each hidden state put trough something to a small nn.
        predictions_logits = self.predictions(out.squeeze(dim=0))

        out_probs = torch.softmax(predictions_logits, dim=-1)

        return predictions_logits, out_probs, hidden.squeeze(dim=0)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
