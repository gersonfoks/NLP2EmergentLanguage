import torch
from torch import nn
import numpy as np

from models.VisualModels import HiddenStateModel


class SenderModelFixedLength(nn.Module):

    def __init__(self, output_dim, msg_len=5, n_symbols=3, hidden_state_model=None, tau=0.5, discreet=True):
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
        self.discreet = discreet

    def forward(self, x):
        hidden_state = self.to_hidden(x)
        output_logits = self.to_msg(hidden_state)
        output_logits = output_logits.reshape(-1, self.msg_len, self.n_symbols)
        msg = torch.nn.functional.gumbel_softmax(output_logits, tau=self.tau, hard=self.discreet, dim=-1)
        return msg


class SenderRnn(nn.Module):
    def __init__(self, output_dim, msg_len=5, n_symbols=3, hidden_state_model=None, tau=1.2):
        '''
        A sender that send fixed length messages
        '''
        super(SenderRnn, self).__init__()
        self.n_symbols = n_symbols
        if hidden_state_model:
            self.to_hidden = hidden_state_model
        else:
            self.to_hidden = HiddenStateModel(output_dim)

        self.hidden_state_size = self.to_hidden.hidden_state_size

        self.gru = nn.LSTM(self.n_symbols, self.hidden_state_size)

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
        cell_state = torch.zeros(1, len(x), self.hidden_state_size).to(hidden_state.device)
        start_symbol = torch.zeros((1, len(x), self.n_symbols)).to(hidden_state.device)
        current_symbol = start_symbol
        result = []

        for i in range(self.msg_len):
            out, (hidden_state, cell_state) = self.gru(current_symbol, (hidden_state, cell_state))

            out = self.to_symbol(out).reshape(1, -1, self.n_symbols)
            symbol = torch.nn.functional.gumbel_softmax(out, tau=self.tau, hard=True, dim=-1)

            result.append(symbol)

        msg = torch.cat(result).permute(1, 0, 2)

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
