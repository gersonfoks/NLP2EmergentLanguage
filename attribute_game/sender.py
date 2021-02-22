import torch
from torch import nn
import numpy as np

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

            nn.Linear(self.feature_encoder.hidden_state_size, self.feature_encoder.hidden_state_size),
            nn.ReLU(),
            nn.Linear(self.feature_encoder.hidden_state_size, n_symbols)
        )

        self.tau = tau
        self.msg_len = msg_len
        self.n_symbols = n_symbols
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

            result.append(out)

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
