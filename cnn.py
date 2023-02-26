import torch
import torch.nn as nn
import torch.nn.functional as F


input_shape_cnn = (1, 6, 12, 3)              #batch_size, num_channels, height, width

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, kernel_size = 3, stride = 1, padding = 'valid'):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(groups = in_channels, in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False) 
        self.pointwise_conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, padding = padding, bias = False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x


class CNN(nn.Module):
    def __init__(self, input_shape=input_shape_cnn, num_classes=10):
        super(CNN, self).__init__()
        # self.dwsp_conv = nn.Conv3d()#DepthwiseSeparableConv(in_channels = 3, out_channels = 128, kernel_size = 3, stride = (1, 1), padding = 'valid')
        # self.act0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels = 10, kernel_size = 3, stride = (1, 1), padding = 1)
        # self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = (3, 1))
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 15, kernel_size = 5, stride = (1, 1), padding = 2)
        # self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = (4, 3))
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(2304, 32)
        # self.fc2 = nn.Linear(32, 64)
        # self.fc3 = nn.Linear(64, 10)
        # self.act_out = nn.Softmax(dim=1)
        # self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 0, dilation = 1)
        # self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 0, dilation = 1)
        # self.conv3 = nn.Conv2d(in_channels = 3, out_channels = 5, kernel_size = 3, stride = 1, padding = 'valid')
        # self.conv4 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 1, padding = 1)
        # self.conv5 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, stride = 2, padding = 1)
    def forward(self, x):
        # x = self.dwsp_conv(x)
        # x = self.act0(x)
        x = self.conv1(x)
        # x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        # x = self.act2(x)
        x = self.pool2(x)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)

        # x = self.conv1(x)
        # x = self.conv2(x)

        return x