import torch.nn as nn
import torch
import MLP

class Decoder(nn.Module):
    def __init__(self, token_dim, num_heads, dropout):
        nn.Module.__init__(self)
        self.self_att1 = nn.MultiheadAttention(token_dim, 4, dropout)
        self.ln1 = nn.LayerNorm((16 * 16, token_dim))
        self.token_dim = token_dim
        self.conv1 = nn.
    
    def forward(self, input):
        '''
            decode the embedded features into image
            input: embedded feature, [bsz, 16 * 16, token_dim]
        '''
        att_output, _ = self.self_att1(input, input)
        res_output = att_output + input
        ln1_output = self.ln1(res_output)
        
        channel_output = ln1_output.transpose(1, 2).view(-1, self.token_dim, 16, 16)
        
        
        
    