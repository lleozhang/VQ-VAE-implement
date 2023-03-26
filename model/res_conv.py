import torch.nn as nn
import torch

class Res_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_normalization = True, 
                 activation = 'relu', *args):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels= in_channels, out_channels= out_channels, 
                              kernel_size= kernel_size, padding= (kernel_size[0] - 1) // 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if with_normalization:
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        
        if activation == 'relu':
            self.act = nn.ReLU(args)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(args)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'tanh':
            self.act = nn.Tanh(args)
        elif activation == 'elu':
            self.act = nn.ELU(args)
        else:
            self.act = None
        
                
        
    def forward(self, input):
        output = self.conv(input)
        
        if self.norm:
            output = self.norm(output)
        
        if self.act:
            output = self.act(output)
        
        if self.in_channels == self.out_channels:
            return input + output #res connection
        else:
            return output