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
from torchsummary import summary

import _init_paths
from config import config
from config import update_config
from dataset import BaseDataset
from unet import UNet
from unet import ResNet50, Lateral, DepthModel
from core.criterion import DepthLoss
from core.functions import train, validate
from utils.utils import FullModel

dataset_config_file_path = 'config_files/dataset_config.yaml'
model_config_file_path = 'config_files/model_config.yaml'
training_config_file_path = 'config_files/training_config.yaml'


def parse_args():
    update_config(config, model_config_file_path, dataset_config_file_path, training_config_file_path)
    
def main():

    parse_args()


    writer_dict = {
        'writer': SummaryWriter("/home/raluca/Monocular_depth_estimation/log/incercari"),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }


    train_dataset = BaseDataset(root = '/home/raluca/data/NYUv2/',
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
    model = ResNet50()

    model = Lateral(ResNet50, (480,640))
    model = DepthModel((480,640))
    summary(model, (3,480,640))
    # for i, batch in enumerate(trainloader):
    #     images, labels, _, _ = batch
    #     out_logit, out_softmax = model(images)

    #     print(out_logit)

    #     break



if __name__ == '__main__':
    main()