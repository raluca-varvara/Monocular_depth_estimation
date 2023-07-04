import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import os
import pprint
import shutil
import sys
import yaml

import logging
import time
import timeit
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from operator import itemgetter

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
from unet import ResNet50, Lateral, DepthModel
from core.criterion import DepthLoss
from core.functions import train, validate, test
from utils.utils import FullModel, create_logger

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


DATASET_CFG = 'config_files/dataset_config.yaml'
MODEL_CFG = 'config_files/model_config.yaml'
TRAINING_CFG = 'config_files/training_config.yaml'

SEP = "########################################################################################"


def parse_args():

    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument("--expname", type=str, default="")
    args = parser.parse_args()
    
    update_config(config, MODEL_CFG, DATASET_CFG, TRAINING_CFG)
    
    return args
def main():
    args = parse_args()

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError('It needs a model to test')

    print('=> loading model from {}'.format(model_state_file))

    model_file = config.TEST.MODEL_FILE
    model_file_dir_list = model_file.split("/")[:-2]
    model_test_dir = Path('/'.join(model_file_dir_list))
    model_test_dir = model_test_dir / "test_images"
    model_test_dir.mkdir(parents=True, exist_ok=True)

    model_test_images = model_test_dir / "images"
    model_test_predictions = model_test_dir / "predictions"
    model_test_images.mkdir(parents=True, exist_ok=True)
    model_test_predictions.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu')

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = DepthModel((550,764))
    # model = UNet(n_channels = 3)
    criterion = DepthLoss(device)
    model = FullModel(model, criterion)

    dump_input = torch.rand(
        (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )

    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        print(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()

    image = cv2.imread('/home/raluca/Monocular_depth_estimation/paintings/painting1.png',cv2.IMREAD_COLOR)

    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= [0.487, 0.417, 0.400]
    image /= [0.288, 0.294, 0.307]
    image = image.transpose((2, 0, 1))
    image = image[None, :, :, :]
    image = torch.from_numpy(image)
    with torch.no_grad():
        depth_or, depth_softmax = model.model(image)
        depth = torch.squeeze(depth_or).cpu().numpy()
        depth = depth * 255
    cv2.imwrite('/home/raluca/Monocular_depth_estimation/paintings/depth_painting1.png', depth)

if __name__ == '__main__':
    main()