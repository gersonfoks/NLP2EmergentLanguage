from torch import nn
import torch

import numpy as np


class ReceiverCombined(nn.Module):

    def __init__(self, hidden_model, rnn_model, n_xs, ):
        super(ReceiverCombined, self).__init__()
        self.hidden_model = hidden_model
        self.rnn_model = rnn_model
        self.hidden_state_size = hidden_model.hidden_state_size

        self.n_xs = n_xs

        self.to_prediction = nn.Sequential(
            nn.Linear(self.hidden_model.hidden_state_size * n_xs + self.rnn_model.hidden_size, 128),
            # Look at the msg and all the hidden states side by side.
            nn.ReLU(),
            nn.Linear(128, n_xs)
        )

    def forward(self, xs, msg):
        hidden_states = []
        for x in xs:
            hidden_state = self.hidden_model(x).unsqueeze(1)
            hidden_states.append(hidden_state)

        hidden = torch.cat(hidden_states, dim=1)
        hidden_states_imgs = hidden.reshape(-1, self.hidden_state_size * self.n_xs)

        ### Put it one after the other
        prediction_logits, prediction_probs, hidden_msg = self.rnn_model(msg)

        ###Get the predictions and the hidden state

        hidden_concat = torch.cat([hidden_states_imgs, hidden_msg], dim=-1)

        out = self.to_prediction(hidden_concat)

        out_probs = torch.softmax(out, dim=-1)

        return out, out_probs, prediction_logits, prediction_probs, hidden_msg
