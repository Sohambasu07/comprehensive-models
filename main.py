import torch
from torchinfo import summary as summary
from torchsummary import summary as torchsumm
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
# model_in_use = 'lstm'
model_in_use = 'transformer'


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
    model = RNNModel(input_shape=input_shape_rnn).to(device)

elif model_in_use is 'lstm':
    model = LSTMModel(input_shape=input_shape_lstm,).to(device)

elif model_in_use is 'transformer':
    model = TransformerModel(enc_tensor_shape = enc_tensor_shape, dec_tensor_shape = dec_tensor_shape).to(device)


if model_in_use is 'cnn':
    summary(model, input_shape_cnn,
            col_names = ["input_size",
                        "output_size",
                        "num_params",
                        # "params_percent",
                        "kernel_size",
                        "mult_adds",
                        # "trainable"
                        ])

elif model_in_use is 'cnn1d':
    summary(model, input_shape_cnn1d,
            col_names = ["input_size",
                        "output_size",
                        "num_params",
                        # "params_percent",
                        "kernel_size",
                        "mult_adds",
                        # "trainable"
                        ])

elif model_in_use is 'transformer':
    summary(model, [enc_tensor_shape, dec_tensor_shape],
            col_names = ["input_size",
                        "output_size",
                        "num_params",
                        # "params_percent",
                        "kernel_size",
                        "mult_adds",
                        # "trainable"
                        ])

elif model_in_use is 'lstm':
    summary(model, input_shape_lstm,
            col_names = ["input_size",
                        "output_size",
                        "num_params",
                        # "params_percent",
                        "kernel_size",
                        "mult_adds",
                        # "trainable"
                        ])

elif model_in_use is 'rnn':
    summary(model, input_shape_rnn,
            col_names = ["input_size",
                        "output_size",
                        "num_params",
                        # "params_percent",
                        "kernel_size",
                        "mult_adds",
                        # "trainable"
                        ])

else :
    summary(model, input_shape,
            col_names = ["input_size",
                        "output_size",
                        "num_params",
                        # "params_percent",
                        "kernel_size",
                        "mult_adds",
                        # "trainable"
                        ])


#For MLP, MLP_with _res and DNN:


#For CNN


#For CNN1D


#For RNN


#For LSTM


#For Transformer

# enc_tensor = torch.Tensor(np.array([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]]))
# dec_tensor = torch.Tensor(np.array([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], []]]))

# print("Encoder Tensor: ", enc_tensor, enc_tensor.shape)
# print("Decoder Tensor: ", dec_tensor, dec_tensor.shape)

# transformer_output, attention = model(enc_tensor, dec_tensor)

# print