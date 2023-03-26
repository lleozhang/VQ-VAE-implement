import torch.nn as nn
import torch

def fea2cha(input, feature_dim, siz):
    '''
        convert a feature representation to a channel map
        input: [bsz, patch_numer, feature_dim]
        output: [bsz, feature_dim, H, W](H * W = patch_number)
    '''
    return input.transpose(1, 2).view(-1, feature_dim, siz, siz)

def cha2fea(input, feature_dim, siz):
    '''
        convert a channel map to a feature representation
        input: [bsz, C, H, W]
        output: [bsz, H*W, C]
    '''
    return input.transpose(1, 2).transpose(2, 3).view(-1, siz * siz, feature_dim)
