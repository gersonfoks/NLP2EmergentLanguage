from torch import nn
import torch

import numpy as np


class HiddenStateModel(nn.Module):
    def __init__(self, output_dim, input_channels=1,):
        '''
        A simple hidden state model that is used to determine the hidden state of a sender/receiver
        Can be pretrained on the labels with the use of "to_predictions" function
        :param output_dim: dimension of the output of the model
        '''
        super(HiddenStateModel, self).__init__()

        self.to_hidden = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(1),
            nn.Linear(9216, 128),

        )

        self.to_prediction = nn.Sequential(
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(128, output_dim)
        )

        self.hidden_state_size = 128

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


class SenderModelFixedLength(nn.Module):

    def __init__(self, output_dim, msg_len=5, n_symbols=3, hidden_state_model=None, tau=0.5):
        '''
        A sender that send fixed length messages
        '''
        super(SenderModelFixedLength, self).__init__()

        if hidden_state_model:
            self.to_hidden = hidden_state_model
        else:
            self.to_hidden = HiddenStateModel(output_dim)

        self.to_msg = nn.Sequential(
            nn.Linear(self.to_hidden.hidden_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, msg_len * n_symbols)
        )

        self.tau = tau
        self.msg_len = msg_len
        self.n_symbols = n_symbols

    def forward(self, x):
        hidden_state = self.to_hidden(x)
        output_logits = self.to_msg(hidden_state)
        output_logits = output_logits.reshape(-1, self.msg_len, self.n_symbols)
        msg = torch.nn.functional.gumbel_softmax(output_logits, tau=self.tau, hard=True, dim=-1)
        return msg


class ReceiverModuleFixedLength(nn.Module):
    def __init__(self, output_dim, msg_len=5, n_symbols=3, n_xs=3, hidden_state_model=None, tau=0.5):
        '''
        A receiver that receives fixed length messages
        '''
        super(ReceiverModuleFixedLength, self).__init__()

        if hidden_state_model:
            self.to_hidden = hidden_state_model
        else:
            self.to_hidden = HiddenStateModel(output_dim)

        self.to_prediction = nn.Sequential(
            nn.Linear(self.to_hidden.hidden_state_size * n_xs + msg_len * n_symbols, 128),
            # Look at the msg and all the hidden states side by side.
            nn.ReLU(),
            nn.Linear(128, n_xs)
        )
        self.tau = tau
        self.hidden_state_size = self.to_hidden.hidden_state_size
        self.n_xs = n_xs
        self.msg_len = msg_len
        self.n_symbols = n_symbols

    def forward(self, xs, msg):
        hidden_states = []
        for x in xs:
            hidden_state = self.to_hidden(x).unsqueeze(1)

            hidden_states.append(hidden_state)

        hidden = torch.cat(hidden_states, dim=1)
        hidden = hidden.reshape(-1, self.hidden_state_size * self.n_xs)

        ### Put it one after the other
        msg = msg.reshape(-1, self.msg_len * self.n_symbols)

        hidden = torch.cat([hidden, msg], dim=-1)

        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs


class PredictionRNN(nn.Module):
    def __init__(self, n_words, hidden_size):
        super(PredictionRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(n_words, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.predictions = nn.Linear(hidden_size, n_words)
        self.n_words = n_words

    def forward(self, input):
        batch_size = input.shape[0]
        input = input.reshape(-1, self.n_words)

        embedded = self.embedding(input)
        embedded = embedded.reshape(batch_size, -1, self.hidden_size)

        out, hidden = self.gru(embedded)

        # Each hidden state put trough something to a small nn.
        predictions_logits = self.predictions(out)

        out_probs = torch.softmax(predictions_logits, dim=-1)
        return predictions_logits, out_probs, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class SenderRnn(nn.Module):
    def __init__(self, output_dim, msg_len=5, n_symbols=3, hidden_state_model=None, tau=0.5):
        '''
        A sender that send fixed length messages
        '''
        super(SenderRnn, self).__init__()

        if hidden_state_model:
            self.to_hidden = hidden_state_model
        else:
            self.to_hidden = HiddenStateModel(output_dim)

        self.hidden_state_size = self.to_hidden.hidden_state_size
        self.gru = nn.GRU(self.hidden_state_size, self.hidden_state_size)

        self.to_symbol = nn.Sequential(
            nn.Linear(self.to_hidden.hidden_state_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_symbols)
        )

        self.tau = tau
        self.msg_len = msg_len
        self.n_symbols = n_symbols

    def forward(self, x):

        ###Generate messages of length msg_len. Once the stop symbol (highest number in our alphabet) is generated the rest of the string will be filled with that sign

        hidden_state = self.to_hidden(x)

        hidden_state = hidden_state.unsqueeze(dim=0)

        result = []
        for i in range(self.msg_len):
            out, hidden_state = self.gru(hidden_state)
            out = self.to_symbol(out).reshape(-1, 1, self.n_symbols)

            result.append(out)

        output_logits = torch.cat(result, dim=1)

        msg = torch.nn.functional.gumbel_softmax(output_logits, tau=self.tau, hard=True, dim=-1)

        ##Now we need to the stop words in the msg

        msg = self.add_stop_symbols(msg)

        return msg

    def add_stop_symbols(self, msg):

        ##TODO: fix this bad programming
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        symbol_tensor = torch.argmax(msg, dim=-1)

        stop_symbol_tensor = symbol_tensor == self.n_symbols - 1

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
            mask[start_index:] = True
            return mask

        np_stop_symbol_tensor = np.apply_along_axis(fill_true, 1, np_stop_symbol_tensor)

        mask = torch.tensor(np_stop_symbol_tensor).to(device)

        m = symbol_tensor * ~mask + torch.ones(symbol_tensor.shape).to(device) * (self.n_symbols - 1) * mask
        m = m.long()
        # For each message we then replace all symbols after the stop symbol to a stop symbol

        mask = mask.unsqueeze(dim=-1).repeat(1, 1, self.n_symbols)

        one_hot = torch.nn.functional.one_hot(m)

        msg = msg * ~mask + one_hot * mask
        return msg.float()
