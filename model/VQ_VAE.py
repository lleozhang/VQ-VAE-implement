import torch
import torch.nn as nn
from .Encoder import Encoder
from .Decoder import Decoder

class VQ_VAE(nn.Module):
    def __init__(self, token_size, token_dim, medium_dim, dropout, num_heads):
        '''
            The whole model of VQ-VAE
            token_size: how many tokens we have in total
            token_dim: the dimension of one token feature
            medium_dim: the number of channels while upsampling
            dropout: dropout
            num_heads: the number of heads of multihead attention
        '''
        nn.Module.__init__(self)
        self.encoder = Encoder(token_size, token_dim)
        self.decoder = Decoder(token_dim, medium_dim, num_heads, dropout)
        
    def forward(self, input):
        '''
            input is a batch of images
            input: [bsz, C, H, W], C = 3, H = W = 224
        '''
        enc_fea, enc_loss = self.encoder(input)

        output = self.decoder(enc_fea)
        return output, enc_loss