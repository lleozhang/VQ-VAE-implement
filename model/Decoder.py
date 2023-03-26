import torch.nn as nn
import torch
from .res_conv import Res_Conv
from .utils import fea2cha, cha2fea

class Decoder(nn.Module):
    def __init__(self, token_dim, medium_dim, num_heads, dropout):
        '''
            To decode a feature map into an image
            token_dim: same as Encoder
            medium_dim: the number of channels while upsampling
            num_heads: the number of heads
            dropout: dropout        
        '''
        nn.Module.__init__(self)
        self.self_att = nn.MultiheadAttention(token_dim, num_heads, dropout, batch_first = True)

        self.token_dim = token_dim
        self.medium_dim = medium_dim
        
        self.conv1 = Res_Conv(in_channels= token_dim, out_channels= token_dim, kernel_size= (5, 5))
        self.upsam1 = nn.Upsample((32, 32), mode = 'bilinear')
        self.self_att1 = nn.MultiheadAttention(token_dim, num_heads, dropout, batch_first = True)
        
        self.conv2 = Res_Conv(in_channels= token_dim, out_channels= medium_dim, kernel_size= (5, 5))
        self.upsam2 = nn.Upsample((64, 64), mode = 'bilinear')
        self.self_att2 = nn.MultiheadAttention(medium_dim, num_heads, dropout, batch_first = True)
        
        self.conv3 = Res_Conv(in_channels= medium_dim, out_channels= medium_dim, kernel_size= (3, 3))
        self.upsam3 = nn.Upsample((128, 128), mode = 'bilinear')
        self.self_att3 = nn.MultiheadAttention(medium_dim, num_heads, dropout, batch_first = True)
        
        self.conv4 = Res_Conv(in_channels= medium_dim, out_channels= medium_dim, kernel_size= (3, 3))
        self.upsam4 = nn.Upsample((224, 224), mode = 'bilinear')
        self.self_att4 = nn.MultiheadAttention(medium_dim, num_heads, dropout, batch_first = True)
        
        self.conv5 = Res_Conv(in_channels= medium_dim, out_channels= 3, kernel_size= (3, 3))
        self.conv6 = Res_Conv(in_channels= 3, out_channels= 3, kernel_size= (3, 3), activation= 'sigmoid')
        
    def forward(self, input):
        '''
            decode the embedded features into image
            input: embedded feature, [bsz, 16 * 16, token_dim]
        '''
        att_output, _ = self.self_att(input, input, input)
        res_output = att_output + input
        
        channel_output = fea2cha(res_output, self.token_dim, 16)
        #[bsz, 1280, 16, 16]
        
        conv1_out = self.conv1(channel_output)
        #[bsz, 500, 16, 16]
        
        upsam1 = self.upsam1(conv1_out)
        #[bsz, 500, 32, 32]        
        
        re_upsam1 = cha2fea(upsam1, self.token_dim, 32)
        att_out1, _ = self.self_att1(re_upsam1, re_upsam1, re_upsam1)
        re_att_out1 = fea2cha((re_upsam1 + att_out1), self.token_dim, 32)
        
        conv2 = self.conv2(re_att_out1)
        upsam2 = self.upsam2(conv2)
        
        re_upsam2 = cha2fea(upsam2, self.medium_dim, 64)
        att_out2, _ = self.self_att2(re_upsam2, re_upsam2, re_upsam2)
        re_att_out2 = fea2cha(att_out2, self.medium_dim, 64) + upsam2
        
        conv3 = self.conv3(re_att_out2)
        upsam3 = self.upsam3(conv3)
    
        re_upsam3 = cha2fea(upsam3, self.medium_dim, 128)
        att_out3, _ = self.self_att3(re_upsam3, re_upsam3, re_upsam3)
        re_att_out3 = fea2cha(att_out3, self.medium_dim, 128) + upsam3

        conv4 = self.conv4(re_att_out3)
        upsam4 = self.upsam4(conv4)
        
        re_upsam4 = cha2fea(upsam4, self.medium_dim, 224)
        att_out4, _ = self.self_att4(re_upsam4, re_upsam4, re_upsam4)
        re_att_out4 = fea2cha(att_out4, self.medium_dim, 224) + upsam4

        conv5 = self.conv5(re_att_out4)
        conv6 = self.conv6(conv5)
        
        return conv6
        
        
        
        
    