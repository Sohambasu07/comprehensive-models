import torch
import torch.nn as nn
import torch.nn.functional as F


input_shape_lstm = (4, 256)         #(timesteps or sequence length, input features)

class LSTMModel(nn.Module):
    def __init__(self, input_shape=input_shape_lstm):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size = 256, hidden_size = 512, num_layers = 1, bias = False)
        
    def forward(self, x):
        x = self.lstm1(x)

        return x