import torch.nn as nn
import torch
import MLP

class Decoder(nn.Module):
    def __init__(self, dim, dropout):
        nn.Module.__init__(self)
        self.self_att1 = nn.MultiheadAttention(dim, 4, dropout)
        
    def forward(self, input):
        att_output, _ = self.self_att1(input, input)
    