import torch
import torch.nn as nn
from torchvision.models import vit_h_14, ViT_H_14_Weights
from .utils import fea2cha, cha2fea

class Encoder(nn.Module):
    def __init__(self, token_size, token_dim = 1280, beta = 0.25):
        '''
            To encode the image into discrete tokens.
            token_size: The number of all tokens.
            token_dim: The dimension of the embedding feature.
            beta: The value of commitment loss
        '''
        nn.Module.__init__(self)
        self.token_size = token_size
        self.token_dim = token_dim
        self.beta = beta
        self.encoder = vit_h_14(weights = ViT_H_14_Weights.DEFAULT)
        self.li = nn.Linear(1280, token_dim)
        self.loss = nn.MSELoss()
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
        img_feature = self.li(img_feature)
        img_quan = img_feature.view(-1, self.token_dim)
        
        #similarity between image and embedding tokens
        Sim =((img_quan ** 2).sum(dim = -1).view(-1, 1) 
              - 2 * img_quan @ (self.embedding.transpose(0, 1))
              + (self.embedding ** 2).sum(dim = -1).view(1, -1))
        
        embedded_token = torch.argmin(Sim, dim = -1)
        embedded_feature = self.embedding[embedded_token]
        
        
        embedded_loss = (self.loss(img_quan.detach(), embedded_feature) #VQ Loss
                        + self.beta * self.loss(embedded_feature.detach(), img_quan)) #Commitment Loss
        embedded_feature = embedded_feature.view(-1, 16 * 16, self.token_dim)
        
        #Loss backward
        embedded_feature = img_feature + (embedded_feature - img_feature).detach()

        return embedded_feature, embedded_loss
    
        
        