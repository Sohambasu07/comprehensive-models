import torch
import torch.nn as nn
import torch.nn.functional as F


input_shape = (4, 2)

class MLP_with_res(nn.Module):
    def __init__(self, input_shape=input_shape, num_classes=2):
        super(MLP_with_res, self).__init__()
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

        # print(self.fc1.weight.shape)
        # print(self.fc2.weight.shape)
        # print(self.fc3.weight.shape)

        print(self.fc1.weight.grad)

        
    def forward(self, x):
        z0 = self.fc1(x)
        h0 = self.act1(z0)
        z1 = self.fc2(h0)
        h1 = self.act2(z1)
        z2 = self.fc3(h1+h0)
        y = self.act3(z2)

        return y