import torch
from torch import nn


class ReceiverRnn(nn.Module):
    def __init__(self, feature_encoder, n_xs, n_symbols=3, msg_len=3 ):
        '''
        A sender that send fixed length messages
        '''
        super(ReceiverRnn, self).__init__()
        self.feature_encoder = feature_encoder
        self.n_symbols = n_symbols
        self.n_xs = n_xs
        self.relu = nn.ReLU()
        self.hidden_state_size = self.feature_encoder.hidden_state_size

        self.embedding_layer = nn.Sequential( nn.Linear(self.n_symbols, self.hidden_state_size))

        self.rnn = RNN(self.hidden_state_size, self.hidden_state_size, self.hidden_state_size, )

        self.to_prediction = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_state_size * (self.n_xs + 1), self.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, self.n_xs)
        )

    def forward(self, xs, msg):

        hidden_states = []
        for x in xs:
            hidden_state = self.feature_encoder(x)

            hidden_states.append(hidden_state)

        # Permute the msg to make sure that the batch is second
        hidden = self.rnn.initHidden(msg.shape[1]).to(msg.device)

        for m in msg:
            next_m = self.embedding_layer(m)

            out, hidden = self.rnn(next_m, hidden)
            hidden= self.relu(hidden)






        hidden_states.append(hidden)

        hidden = torch.cat(hidden_states, dim=1)

        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)
class ReceiverLSTM(nn.Module):
    def __init__(self, feature_encoder, n_xs, n_symbols=3, msg_len=3 ):
        '''
        A sender that send fixed length messages
        '''
        super(ReceiverLSTM, self).__init__()
        self.feature_encoder = feature_encoder
        self.n_symbols = n_symbols
        self.n_xs = n_xs
        self.relu = nn.ReLU()
        self.hidden_state_size = self.feature_encoder.hidden_state_size

        self.embedding_layer = nn.Sequential( nn.Linear(self.n_symbols, self.hidden_state_size))

        self.rnn = nn.LSTM(self.n_symbols, self.hidden_state_size )

        self.to_prediction = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_state_size * (self.n_xs + 1), self.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, self.n_xs)
        )

    def forward(self, xs, msg):


        hidden_states = []
        for x in xs:
            hidden_state = self.feature_encoder(x)

            hidden_states.append(hidden_state)

        # Permute the msg to make sure that the batch is second

        #msg = self.embedding_layer(msg.view(-1, self.n_symbols))

        out, hidden = self.rnn(msg)

        hidden = self.relu(hidden[0][0])


        hidden_states.append(hidden)

        hidden = torch.cat(hidden_states, dim=1)

        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs

