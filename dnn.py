import torch
import torch.nn as nn
import torch.nn.functional as F


input_shape = (4, 2)

class DNN(nn.Module):
    def __init__(self, input_shape=input_shape, num_classes=2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_shape[1], 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 4)
        self.fc5 = nn.Linear(4, 4)
        self.fc6 = nn.Linear(4, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        return x