import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias = True, 
                normoalization = True, activation = 'relu', *args):
        nn.Module.__init__(self)
        self.li = nn.Linear(input_dim, output_dim, bias = bias)
        if normoalization:
            self.norm = nn.BatchNorm1d(output_dim)
        else:
            self.norm = None

        if activation == 'relu':
            self.act = nn.ReLU(args)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid(args)
        elif activation == 'tanh':
            self.act = nn.Tanh(args)
        elif activation == 'softmax':
            self.act = nn.Softmax(args)
        elif activation == 'elu':
            self.act = nn.ELU(args)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(args)
        else:
            self.act = None
        
        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        output = self.li(input)
        if self.norm:
            output = self.norm(output)
        if self.act:
            output = self.act(output)
        output = self.drop(output)
        return output
