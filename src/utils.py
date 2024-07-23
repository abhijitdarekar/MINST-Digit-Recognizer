import torch
from torch import nn
from torch import functional as F

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
