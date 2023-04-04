import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor, Normalize
from PIL import Image
import os



class ImageSet(Dataset):
    def __init__(self, path):
        Dataset.__init__(self)
        
        self.path = path
        self.img_lis = os.listdir(path)
        self.len = len(self.img_lis)
        self.transform = Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                #Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ]
        )
        
    def __getitem__(self, index):
        img = Image.open(self.path + self.img_lis[0]).convert('RGB')
        return self.transform(img)
    
    def __len__(self):
        return self.len