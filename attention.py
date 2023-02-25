import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

x = torch.Tensor(np.array([[1, 1], [1, 1], [0, 0]]))  
W_Q = torch.Tensor(np.array([[0.0], [1.0]]))  
W_K = torch.Tensor(np.array([[1.0], [0.0]]))  
W_V = torch.Tensor(np.array([[0.5], [0.5]]))
W_O = torch.Tensor(np.array([[0.5, 0.5]]))

Q = torch.matmul(x, W_Q)
K = torch.matmul(x, W_K)
V = torch.matmul(x, W_V)

print("Query:",Q)
print("Key:",K)
print("Value:",V)

mult_Q_K = torch.matmul(Q, torch.transpose(K, 0, 1))
Attention_weights = F.softmax(mult_Q_K, dim=1)
Attention = torch.matmul(Attention_weights, V)
Output = Attention@W_O
print(mult_Q_K)
print(Attention_weights)
print(Attention)
print(Output)
