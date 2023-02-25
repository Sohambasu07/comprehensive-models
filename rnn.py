import torch
import torch.nn as nn
import torch.nn.functional as F


input_shape_rnn = (4, 256)         #(timesteps or sequence length, input features)

class RNNModel(nn.Module):
    def __init__(self, input_shape=input_shape_rnn):
        super(RNNModel, self).__init__()
        self.lstm1 = nn.RNN(input_size = 256, hidden_size = 512, num_layers = 1, nonlinearity = 'tanh', bias = False, bidirectional = False)
        
    def forward(self, x):
        x = self.lstm1(x)

        return x