import torch.nn as nn
import torch

class Res_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_normalization, 
                 activation = 'relu', *args):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels= out_channels, 
                              kernel_size= kernel_size, padding= (kernel_size - 1) // 2)
        if self.with_normalization:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        
        if self.activation == 'relu':
            self.act = nn.ReLU(args)
        elif self.activation == 'leakyrelu':
            self.act = nn.LeakyReLU(args)
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid(args)
        elif self.activation == 'tanh':
            self.act = nn.Tanh(args)
        elif self.activation == 'elu':
            self.act = nn.ELU(args)
        else:
            self.act = None
        
                
        
    def forward(self, input):
        output = self.conv(input)
        
        if self.norm:
            output = self.norm(output)
        
        if self.act:
            output = self.act(output)
            
        return input + output #res connection