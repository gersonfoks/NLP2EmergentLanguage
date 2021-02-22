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


class SenderRnn(nn.Module):
    def __init__(self, feature_encoder, msg_len=5, n_symbols=3, tau=0.8):
        '''
        A sender that send fixed length messages
        '''
        super(SenderRnn, self).__init__()
        self.feature_encoder = feature_encoder
        self.n_symbols = n_symbols

        self.hidden_state_size = self.feature_encoder.hidden_state_size

        self.rnn = nn.LSTM(self.n_symbols, self.hidden_state_size)

        self.to_symbol = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.feature_encoder.hidden_state_size, self.feature_encoder.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.feature_encoder.hidden_state_size, n_symbols)
        )

        self.tau = tau
        self.msg_len = msg_len
        self.n_symbols = n_symbols

    def forward(self, x):

        ###Generate messages of length msg_len. Once the stop symbol (highest number in our alphabet) is generated the rest of the string will be filled with that sign

        hidden_state = self.feature_encoder(x)

        hidden_state = hidden_state.view(1, len(x), -1)
        cell_state = torch.zeros(1, len(x), self.hidden_state_size).to(hidden_state.device)
        start_symbol = torch.zeros((1, len(x), self.n_symbols)).to(hidden_state.device)
        hidden_state = (hidden_state, cell_state)
        current_symbol = start_symbol
        result = []

        for i in range(self.msg_len):
            out, hidden_state = self.rnn(current_symbol, hidden_state)
            out = out.view(-1, self.feature_encoder.hidden_state_size)
            out = self.to_symbol(out).view(1, -1, self.n_symbols)

            symbol = torch.nn.functional.gumbel_softmax(out, tau=self.tau, hard=True, dim=-1)
            current_symbol = symbol

            result.append(symbol)

        msg = torch.cat(result)


        msg = self.add_stop_symbols(msg)



        return msg

    def add_stop_symbols(self, msg):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### Get the symbols
        symbol_tensor = torch.argmax(msg, dim=-1)

        ### Get tensor which has true whenever there is a stop symbol
        stop_symbol_tensor = symbol_tensor == 0

        ### Move it to numpy
        np_stop_symbol_tensor = stop_symbol_tensor.cpu().numpy()

        def fill_true(mask, ):
            start_index = 0

            for i, s in enumerate(mask):
                if i == len(mask) - 1:
                    break
                if s:
                    start_index = i
                    break
            mask[start_index] = False
            if start_index < len(mask):
                start_index += 1
            mask[start_index + 1:] = True
            return mask

        ###Once we pack from the first true onward
        np_stop_symbol_tensor = np.apply_along_axis(fill_true, 0, np_stop_symbol_tensor)

        ### Get the masks
        mask = torch.tensor(np_stop_symbol_tensor).to(device)

        m = symbol_tensor * ~mask + torch.ones(symbol_tensor.shape).to(device) * (self.n_symbols - 1) * mask
        m = m.long()

        ### We not create it unto the one hot encoding with the mask
        mask = mask.unsqueeze(dim=-1).repeat(1, 1, self.n_symbols)

        one_hot = torch.nn.functional.one_hot(m, num_classes=self.n_symbols)

        msg = msg * ~mask + one_hot * mask

        # Make sure it is off the right type
        msg = msg.float()

        return msg




class SenderFixed(nn.Module):
    def __init__(self, feature_encoder, msg_len=5, n_symbols=3, tau=0.8):
        '''
        A sender that send fixed length messages
        '''
        super(SenderFixed, self).__init__()
        self.feature_encoder = feature_encoder
        self.n_symbols = n_symbols

        self.hidden_state_size = self.feature_encoder.hidden_state_size

        self.to_msg = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_state_size, msg_len * n_symbols)
        )

        self.tau = tau
        self.msg_len = msg_len
        self.n_symbols = n_symbols

    def forward(self, x):
        ###Generate messages of length msg_len. Once the stop symbol (highest number in our alphabet) is generated the rest of the string will be filled with that sign

        hidden_state = self.feature_encoder(x)
        msg_logits = self.to_msg(hidden_state)
        msg_logits = msg_logits.reshape(self.msg_len, len(x), self.n_symbols)
        msg = torch.nn.functional.gumbel_softmax(msg_logits, tau=self.tau, hard=True, dim=-1)



        return msg


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

            nn.ReLU(),
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
