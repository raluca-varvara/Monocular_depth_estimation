import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import _init_paths
from config import config
from config import update_config
from dataset import BaseDataset
from unet import UNet
from core.criterion import DepthLoss
from core.functions import train, validate
from utils.utils import FullModel

dataset_config_file_path = '/Users/varvararaluca/Documents/Facultate/LICENTA/UNET_monocular_depth_estimation/config_files/dataset_config.yaml'
model_config_file_path = '/Users/varvararaluca/Documents/Facultate/LICENTA/UNET_monocular_depth_estimation/config_files/model.config.yaml'
training_config_file_path = '/Users/varvararaluca/Documents/Facultate/LICENTA/UNET_monocular_depth_estimation/config_files/training_config.yaml'

def parse_args():
    update_config(config, model_config_file_path, dataset_config_file_path, training_config_file_path)
    
def main():

    parse_args()


    writer_dict = {
        'writer': SummaryWriter("/Users/varvararaluca/Documents/Facultate/LICENTA/UNET_monocular_depth_estimation/outputs/tensorboards"),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }


    train_dataset = BaseDataset(root = '/Users/varvararaluca/Documents/Datasets/nyu_data/',
                                list_path = 'metadata.csv',
                                base_size = config.TRAIN.BASE_SIZE,
                                crop_size= config.TRAIN.IMAGE_SIZE,
                                multiscale = config.TRAIN.MULTI_SCALE,
                                multiscale_factor = config.TRAIN.MULTI_SCALE_FACTOR)
    print(config)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        )
    print(len(trainloader))


if __name__ == '__main__':
    main()