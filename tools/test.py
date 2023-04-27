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


    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = DepthModel((480,640))
    criterion = DepthLoss()
    model = FullModel(model, criterion)

    dump_input = torch.rand(
        (1, 3, config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )

    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))


        
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        print(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # gpus = list(config.GPUS)
    # model = nn.DataParallel(model, device_ids=gpus).cuda()
    test_dataset = BaseDataset(root = config.DATASET.ROOT,
                            list_path = 'metadata.csv',
                            stage = 'test',
                            num_samples = 10,
                            base_size = config.TEST.BASE_SIZE,
                            crop_size= config.TEST.IMAGE_SIZE)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    test_loss, test_mse, test_mae, test_abs_rel, test_delta1, test_delta2, test_delta3, test_log10 = test(config, testloader, model, model_test_dir)
    
    # start = timeit.default_timer()
    # #if 'val' in config.DATASET.TEST_SET:
    # mean_IoU, IoU_array, pixel_acc, mean_acc = testval(config, 
    #                                                        test_dataset, 
    #                                                        testloader, 
    #                                                        model,
    #                                                        sv_dir=final_output_dir,
    #                                                        sv_pred = False)
    
    # msg = 'MeanIU: {: 4.4f}, Pixel_Acc: {: 4.4f}, \
    #         Mean_Acc: {: 4.4f}, Class IoU: '.format(mean_IoU, 
    #         pixel_acc, mean_acc)
    # logging.info(msg)
    # logging.info(IoU_array)
    
    # end = timeit.default_timer()
    # #elif 'test' in config.DATASET.TEST_SET:
    # # test(config, 
    # #          test_dataset, 
    # #          testloader, 
    # #          model,
    # #          sv_dir=final_output_dir)

    
    # logger.info('Mins: %d' % np.int((end-start)/60))
    # logger.info('Done')


if __name__ == '__main__':
    main()
