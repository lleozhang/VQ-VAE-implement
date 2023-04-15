import torch
import torch.nn as nn
from torchvision.models import vit_h_14, ViT_H_14_Weights
from .res_conv import Res_Conv
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
        self.encoder = vit_h_14(weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
        self.li = nn.Linear(1280, token_dim)
        self.ln = nn.LayerNorm((16 * 16, 1280))
        self.loss = nn.MSELoss()
        for para in list(self.encoder.parameters()):
            para.require_grad = False
        self.embedding = nn.Embedding(token_size, token_dim)
        self.embedding.weight.data.uniform_(-1/10, 1/10)

        
    def forward(self, img, debug):
        with torch.no_grad():
            img_feature = self.encoder.encoder.layers(
                cha2fea(self.encoder.encoder.dropout(
                    self.encoder.conv_proj(img)
                ), 1280, 16)
            )
        img_feature = self.ln(img_feature)
        
        img_quan = img_feature.view(-1, self.token_dim)
        
        if debug:
            print(img_quan[0])
        
        #similarity between image and embedding tokens
        Sim = torch.sum(img_quan**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(img_quan, self.embedding.weight.t())
        
        embedded_token = torch.argmin(Sim, dim = -1)
        embedded_feature = self.embedding(embedded_token)
        
        
        embedded_loss = self.loss(img_quan.detach(), embedded_feature) \
                        + self.beta * self.loss(embedded_feature.detach(), img_quan) #Commitment Loss
        embedded_feature = embedded_feature.view(-1, 16 * 16, self.token_dim)
        
        #Loss backward
        embedded_feature = img_feature + (embedded_feature - img_feature).detach()
        if debug:
            print(embedded_token)
            
        return embedded_feature, embedded_loss
    
        
        