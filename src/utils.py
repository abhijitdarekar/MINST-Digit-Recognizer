import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from src.config import CONFIG
import numpy as np

# Python Dataset Class to load Data in Dataloader
class CustomDataset(Dataset):
    def __init__(self,data,transform = None):
        self.data = data
        self.transform = transform

    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data.iloc[index]
        
        label = data[0]
        image_data = data[1:].values.astype(np.uint8).reshape(CONFIG['IMAGE_SIZE'])

        if self.transform is not None:
            image_data = self.transform(image_data)

        return image_data,label
    

def get_transforms():
    """
    Returns transformation for train and test dataset.
    
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    """
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
    return (transform_train, transform_test)

def MINSTModel():
    class Model(nn.Module):
        def __init__(self, in_channels =1 ,out_channels = 10):
            super().__init__()
            self.c1 = nn.Conv2d(in_channels=in_channels,out_channels=8,kernel_size=3,stride =1,padding=2)
            self.b1 = nn.BatchNorm2d(8)

            self.c2 = nn.Conv2d(in_channels = 8,out_channels=32,kernel_size=3,padding=2,stride=1)
            self.b2 = nn.BatchNorm2d(32)

            self.c3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=2)
            self.b3 = nn.BatchNorm2d(64)

            self.l1 = nn.Linear(64*5*5,1024)
            self.l2 = nn.Linear(1024,512)

            self.l3 = nn.Linear(512,128)

            self.l4 = nn.Linear(128,32)
            self.final = nn.Linear(32,out_channels)
            

        def forward(self,x):
            x =F.relu(self.b1(self.c1(x)))
            x = F.max_pool2d(x,2,2) # 1,14,14
        
            x = F.relu(self.b2(self.c2(x))) 
            x = F.max_pool2d(x,2,2) #1,8,8
            x = F.dropout(x,0.3)
            x = F.relu(self.b3(self.c3(x)))
            x = F.max_pool2d(x,2,2) # 1,5,5
            x = F.dropout(x,0.4)
            x = F.relu(self.l1(x.reshape(-1,64*5*5)))
            x = F.relu(self.l2(x))
            x = F.dropout(x)
            x = F.relu(self.l3(x))
            x = self.final(F.relu(self.l4(x)))
            return x
    
    return Model
    

