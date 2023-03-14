import torch
import torch.nn as nn
from torchvision.models import vit_h_14, ViT_H_14_Weights
from MLP import MLP

class Encoder(nn.Module):
    def __init__(self, token_size, token_dim = 1280):
        '''
            To encode the image into discrete tokens.
            token_size: The number of all tokens.
            token_dim: The dimension of the embedding feature.
        '''
        nn.Module.__init__(self)
        self.encoder = vit_h_14(weights = ViT_H_14_Weights.DEFAULT)
        for para in self.encoder.parameters():
            para.require_grad = False
        embedding = torch.randn((token_size, token_dim), requires_grad = True)
        self.embedding = nn.Parameter(embedding)
        self.register_parameter('Image Embedding', self.embedding)
        
    def forward(self, img):
        img_feature = self.encoder.encoder.layers(
            self.encoder.encoder.dropout(
                self.encoder.conv_proj(img)
            ).transpose(1, 2).transpose(2, 3).view(-1, 16 * 16, 1280)
        )#[bsz, 16 * 16, 1280]
        
        #similarity between image and embedding tokens
        Sim = torch.matmul(img_feature/img_feature.norm(dim = -1).unsqueeze(2), 
                            (self.embedding/self.embedding.norm(dim = -1).unsqueeze(1)).T)
        embedded_token = torch.argmax(Sim, dim = -1)
        
        embedded_feature = torch.gather(self.embedding.expand(embedded_token.shape[-1], self.embedding.shape[0], self.embedding.shape[1]),
                                        index = embedded_token.expand(self.embedding.shape[-1], embedded_token.shape[0], embedded_token.shape[1]).permute(1, 2, 0))
        return embedded_token, embedded_feature        
        
        