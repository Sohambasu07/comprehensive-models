import torch
import torch.nn as nn
import torch.nn.functional as F


input_shape_cnn1d = (1, 1, 4)           # batch_size, num_channels, feature size

class CNN1D(nn.Module):
    def __init__(self, input_shape=input_shape_cnn1d, num_classes=10):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, (1, 2), 2)
        self.conv1.weight = torch.nn.parameter.Parameter(torch.tensor([[[1.0, 2.0]]]), requires_grad = True)
        self.conv1.bias = torch.nn.parameter.Parameter(torch.tensor([1.0]), requires_grad = True)


    def forward(self, x):
        x = self.conv1(x)

        return x