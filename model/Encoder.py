import torch
import torch.nn as nn
from torchvision.models import vit_h_14, ViT_H_14_Weights
from .utils import fea2cha, cha2fea

class Encoder(nn.Module):
    def __init__(self, token_size, token_dim = 1280):
        '''
            To encode the image into discrete tokens.
            token_size: The number of all tokens.
            token_dim: The dimension of the embedding feature.
        '''
        nn.Module.__init__(self)
        self.encoder = vit_h_14(weights = ViT_H_14_Weights.DEFAULT)
        self.li = nn.Linear(1280, token_dim)
        
        for para in self.encoder.parameters():
            para.require_grad = False
        embedding = torch.randn((token_size, token_dim), requires_grad = False)
        self.embedding = nn.Parameter(embedding)
        self.register_parameter('Image Embedding', self.embedding)
    
        
    def forward(self, img):
        with torch.no_grad():
            img_feature = self.encoder.encoder.layers(
                cha2fea(self.encoder.encoder.dropout(
                    self.encoder.conv_proj(img)
                ), 1280, 16)
            )#[bsz, 16 * 16, 1280]
        self.img_feature = self.li(img_feature)
        
        #similarity between image and embedding tokens
        Sim = torch.matmul(img_feature/img_feature.norm(dim = -1).unsqueeze(2), 
                            (self.embedding/self.embedding.norm(dim = -1).unsqueeze(1)).T)
        embedded_token = torch.argmax(Sim, dim = -1)

        embedded_feature = self.embedding[embedded_token]
        return embedded_token, embedded_feature 
    
    def update(self, grad):
        self.img_feature.grad = grad
        self.img_feature.backward(grad)
        
        