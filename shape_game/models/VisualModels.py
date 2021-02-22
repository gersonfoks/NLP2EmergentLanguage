import torch
from torch import nn


class HiddenStateModel(nn.Module):
    def __init__(self, output_dim, input_channels=1, ):
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


class VisualModel(nn.Module):
    def __init__(self, output_dim, input_channels=1, ):
        '''
        A simple hidden state model that is used to determine the hidden state of a sender/receiver
        Can be pretrained on the labels with the use of "to_predictions" function
        :param output_dim: dimension of the output of the model
        '''
        super(VisualModel, self).__init__()

        self.hidden_state_size = 2048

        self.layers = [
                          VisualLayer(in_channels=3),

                      ] + [
                          VisualLayer(in_channels=20) for i in range(3)
                      ] + [
                          nn.Flatten(),
                          nn.Linear(20, out_features=self.hidden_state_size),

                      ]

        self.to_hidden = nn.Sequential(
            nn.ReLU(),
            *self.layers
        )

        self.to_prediction = nn.Sequential(

            nn.Linear(self.hidden_state_size, output_dim)
        )

    def to_predictions(self, x):
        '''
        Calculates the output of the model given the input
        :param x: input for the model
        :return: torch.tensor: self.all_modules(x)
        '''
        # y = x.copy()
        # for layer in self.layers:
        #     y = layer()
        hidden = self.to_hidden(x)
        out = self.to_prediction(hidden)

        out_probs = torch.softmax(out, dim=-1)
        return out, out_probs

    def forward(self, x):
        return self.to_hidden(x)


class VisualLayer(nn.Module):

    def __init__(self, in_channels, out_channels=20, kernel_size=3, padding=0, stride=2):
        super(VisualLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):

        return self.layer(x)
