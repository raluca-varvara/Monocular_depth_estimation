import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt

device      = torch.device('cpu') 
num_workers = 4
batch_size  = 128

data_path = '/Users/varvararaluca/Documents/Datasets/nyu_data/'

height = 484
width = 840

def get_train_file_path(image_path):
    
    return data_path + image_path

class ImageData(Dataset):
    
    def __init__(self, df, transform):
        super().__init__()
        self.df         = df
        self.file_paths = df['file_path'].values
        self.transform  = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # import
        file_path = self.file_paths[idx]        
        image     = cv2.imread(file_path, cv2.IMREAD_COLOR) 
        if image is None:
            raise FileNotFoundError(file_path)
            
        # augmentations
        if self.transform:
            image = self.transform(image = image)['image']
            
        return image

if __name__ == '__main__':

    df = pd.read_csv(data_path + 'metadata.csv')
    df['file_path'] = df['label_path'].apply(get_train_file_path)

    augs = A.Compose([A.Resize(height  = height, 
                           width   = width),
                  A.Normalize(mean = (0), 
                              std  = (1)),
                  ToTensorV2()])
    # dataset
    image_dataset = ImageData(df        = df, 
                            transform = augs)

    # data loader
    image_loader = DataLoader(image_dataset, 
                            batch_size  = batch_size, 
                            shuffle     = False, 
                            num_workers = num_workers)
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        # print(inputs.shape)
        psum    += inputs.sum(axis        = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3])
    
    count = len(df) * height * width

    # mean and STD
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)
    print("mean", total_mean)
    print("var", total_var)
    print("std", total_std)