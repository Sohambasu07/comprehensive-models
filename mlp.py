import torch
import torch.nn as nn
import torch.nn.functional as F


input_shape = (4, 2)

class MLP(nn.Module):
    def __init__(self, input_shape=input_shape, num_classes=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape[1], 1)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(1, 1)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(1, 1)
        self.act3 = nn.Sigmoid()

        self.fc1.weight = torch.nn.parameter.Parameter(torch.tensor([[0.3, 0.3]]), requires_grad = True)
        self.fc2.weight = torch.nn.parameter.Parameter(torch.Tensor([[0.2]]), requires_grad = True)
        self.fc3.weight = torch.nn.parameter.Parameter(torch.Tensor([[0.1]]), requires_grad = True)

        self.fc1.bias = torch.nn.parameter.Parameter(torch.Tensor([[0.0]]), requires_grad = True)
        self.fc2.bias = torch.nn.parameter.Parameter(torch.Tensor([[0.0]]), requires_grad = True)
        self.fc3.bias = torch.nn.parameter.Parameter(torch.Tensor([[0.0]]), requires_grad = True)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)

        return x