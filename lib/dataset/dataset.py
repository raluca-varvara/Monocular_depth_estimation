import os

import cv2
import numpy as np
import pandas as pd
import random


import torch
from torch.nn import functional as F
from torch.utils import data
from PIL import Image

class BaseDataset(data.Dataset):
    def __init__(self,  
                 root,
                 list_path,
                 num_samples = None, 
                 stage = 'train',
                 mini_train=False,
                 base_size=640,
                 crop_size=(480, 640), 
                 mean=[0.487, 0.417, 0.400], 
                 std=[0.288, 0.294, 0.307],
                 mean_depth=[0.287], 
                 std_depth=[0.154],
                 random_flip = True,
                 multiscale=True,
                 multiscale_factor=1.5):

        self.base_size = base_size
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.multiscale = multiscale
        self.multiscale_factor = multiscale_factor

        self.mean = mean
        self.std = std
        self.mean_depth = mean_depth
        self.std_depth = std_depth

        self.root = root
        self.metadata_csv = root + list_path
        self.mini_train = mini_train

        self.stage = stage
        
        metadata = pd.read_csv(self.metadata_csv, sep = ',', index_col=False)
        self.files = self.get_samples(metadata)
        
        if num_samples:
            self.files = self.files[:num_samples]

    def __len__(self):
        return len(self.files)

    def get_samples(self, metadata):

        # filter by stage
        metadata = metadata[metadata.dataset_split == self.stage]

        # if mini train and training, filter
        if self.mini_train and self.stage == 'train':
            metadata = metadata[metadata.mini_train == True]
        
        # construct image names
        metadata["name"] = metadata.apply(lambda x: os.path.splitext(os.path.basename(x.image_path))[0], axis=1)

        # select only relevant columns
        selected_columns = {'image_path': 'img', 'label_path': 'label', 'name': 'name'}
        metadata = metadata.rename(columns=selected_columns)[selected_columns.values()]

        # construct list of instance dicts and return
        return metadata.to_dict(orient='records')

    def __getitem__(self, index):
        
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(self.root + item["img"],
                           cv2.IMREAD_COLOR)
        label = cv2.imread(self.root + item["label"],
                           cv2.IMREAD_GRAYSCALE)
        
        # Randomly flip the images horizontally
        if self.random_flip and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        
        # if we want to preprocess by multiscaling
        if self.multiscale:
            factor = np.random.random_sample()*(self.multiscale_factor - 1.0) + 1.0
            h, w, _ = image.shape
            ## TODO check if this is good
            new_w = int(factor * w )
            new_h = int(factor * h )
            image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, (new_w, new_h), interpolation = cv2.INTER_LINEAR)

        # random crop
        if image.shape[0] > self.crop_size[1]:
            x_offset = np.random.randint(0, image.shape[0] - self.crop_size[1])
        else:
            x_offset = 0
        if image.shape[1] > self.crop_size[0]:
            y_offset = np.random.randint(0, image.shape[1] - self.crop_size[0])
        else:
            y_offset = 0
        image = image[x_offset:(self.crop_size[1]+x_offset), y_offset:(self.crop_size[0]+y_offset)]
        label = label[x_offset:(self.crop_size[1]+x_offset), y_offset:(self.crop_size[0]+y_offset)]


        size = image.shape

        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name

    def input_transform(self, image):

        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        label = label.astype(np.float32)
        label = label / 255.0
        label -= self.mean_depth
        label /= self.std_depth
        return label
    
    