import torch
import torch.nn as nn
import torch.nn.functional as F


input_shape_rnn = (3, 120)         #(timesteps or sequence length, input features)

class RNNModel(nn.Module):
    def __init__(self, input_shape=input_shape_rnn):
        super(RNNModel, self).__init__()
        self.rnn1 = nn.RNN(input_size = 120, hidden_size = 300, num_layers = 1, nonlinearity = 'tanh', bias = True, bidirectional = False)
        
    def forward(self, x):
        x = self.rnn1(x)

        return x