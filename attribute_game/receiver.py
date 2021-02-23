import torch
from torch import nn


class ReceiverFixed(nn.Module):
    def __init__(self, feature_encoder, n_xs, n_symbols=3, msg_len=5):
        '''
        A sender that send fixed length messages
        '''
        super(ReceiverFixed, self).__init__()
        self.feature_encoder = feature_encoder
        self.n_symbols = n_symbols
        self.n_xs = n_xs

        self.hidden_state_size = self.feature_encoder.hidden_state_size

        self.msg_to_hidden = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_symbols * msg_len, self.hidden_state_size)
        )

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

        msg = msg.view(msg.shape[1], -1)

        hidden_msg = self.msg_to_hidden(msg)
        # Permute back

        hidden_states.append(hidden_msg)

        hidden = torch.cat(hidden_states, dim=1)

        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs


class ReceiverPredictor(nn.Module):
    def __init__(self, feature_encoder, n_xs, n_symbols=3, msg_len=5):
        '''
        A sender that send fixed length messages
        '''
        super(ReceiverPredictor, self).__init__()
        self.feature_encoder = feature_encoder
        self.n_symbols = n_symbols
        self.n_xs = n_xs

        self.hidden_state_size = self.feature_encoder.hidden_state_size

        self.to_prediction = nn.Sequential(

            nn.Linear(self.hidden_state_size * (self.n_xs + 1), self.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, self.n_xs)
        )

    def forward(self, xs, hidden):
        hidden_states = []
        for x in xs:
            hidden_state = self.feature_encoder(x)

            hidden_states.append(hidden_state)

        # Permute the msg to make sure that the batch is second

        # Permute back

        hidden_states.append(hidden)

        hidden = torch.cat(hidden_states, dim=1)

        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs

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

        hidden = hidden[0][0]


        hidden_states.append(hidden)

        hidden = torch.cat(hidden_states, dim=1)

        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs

