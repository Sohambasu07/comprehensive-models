import torch
from torchinfo import summary
import numpy as np
from mlp import *
from mlp_with_res import *
from dnn import *
from cnn import *
from cnn1d import *
from rnn import *
from lstm import *
from transformer import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# model_in_use = 'mlp'
# model_in_use = 'mlp_with_res'
# model_in_use = 'dnn'
# model_in_use = 'cnn'
# model_in_use = 'cnn1d'
# model_in_use = 'rnn'
model_in_use = 'lstm'
# model_in_use = 'transformer'


torch.cuda.manual_seed_all(2)
if model_in_use is 'mlp':
    model = MLP(input_shape=input_shape,
                        num_classes=2).to(device)

elif model_in_use is 'mlp_with_res':
    model = MLP_with_res(input_shape=input_shape,
                        num_classes=2).to(device)

elif model_in_use is 'dnn':
    model = DNN(input_shape=input_shape,
                        num_classes=2).to(device)

elif model_in_use is 'cnn':
    model = CNN(input_shape=input_shape_cnn,
                        num_classes=2).to(device)

elif model_in_use is 'cnn1d':
    model = CNN1D(input_shape=input_shape_cnn1d,
                        num_classes=2).to(device)

elif model_in_use is 'rnn':
    model = RNNModel(input_shape=input_shape_rnn,
                        num_classes=2).to(device)

elif model_in_use is 'lstm':
    model = LSTMModel(input_shape=input_shape_lstm,
                        num_classes=2).to(device)

num_epochs = 1
lr = 0.01
Adam = torch.optim.Adam(model.parameters(), lr)
SGD = torch.optim.SGD(model.parameters(), lr)
criterion = nn.MSELoss()
# print("Device used for training: ", torch.cuda.get_device_name(torch.cuda.current_device()))

#inputs and labels

x = torch.Tensor(np.array([[[1, 2, 3, 4]]]))
y = torch.Tensor(np.array([[[1, 0]]]))
print("Input Tensor: ", x, x.shape)
print("Output Tensor: ", y, y.shape)



for epoch in range(num_epochs):
    # print("Epoch: ", epoch)
    outputs = model(x.cuda())
    print("Predictions: ", outputs, outputs.shape)
    loss = criterion(outputs, y.cuda())
    print("Loss: ", loss)
    # for params in model.parameters():
    #     print(params.data)
    SGD.zero_grad()
    loss.backward()
    SGD.step()
    count = 1
    wob = 1
    for params in model.parameters():
        print("\nLayer {} params".format(count),"\n")
        if wob == 1: print("Weights: ")
        else: print("Biases: ")
        print(params.grad)
        wob += 1


    


