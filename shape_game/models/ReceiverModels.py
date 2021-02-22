import torch
from torch import nn

from shape_game.models.VisualModels import HiddenStateModel


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
