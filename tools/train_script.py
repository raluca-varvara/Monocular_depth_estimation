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
from core.functions import train, validate
from utils.utils import FullModel, create_logger

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


DATASET_CFG = 'config_files/dataset_config.yaml'
MODEL_CFG = 'config_files/model_config.yaml'
TRAINING_CFG = 'config_files/training_config.yaml'

SEP = "########################################################################################"

def parse_args():
    ## modified
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpuid", type=int, required=True)
    parser.add_argument("--expname", type=str, required=True)
    args = parser.parse_args()
    
    update_config(config, MODEL_CFG, DATASET_CFG, TRAINING_CFG)
    
    return args
    
def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir, checkpoints_dir = create_logger(
        config, MODEL_CFG, args.expname,  'train')
    
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    print(SEP)
    print("DISTRIBUTED")
    print(distributed)
    print(SEP)
    device = torch.device('cuda:{}'.format(args.gpuid))
    print("Device: ",device)
    print(f"Training started on GPU: {args.gpuid}")
    print(SEP)

    # dump_input = torch.rand(
    #         (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    #         )
    # logger.info(get_model_summary(model.to(device), dump_input.to(device)))
    config_dump_path = os.path.join(final_output_dir, 'config.yaml')
    with open(config_dump_path, 'w') as outfile:
            yaml.dump(config.dump(), outfile)


    train_dataset = BaseDataset(root = config.DATASET.ROOT,
                                mini_train = config.TRAIN.MINI_SET,
                                num_samples= None,
                                stage = 'train',
                                list_path = 'metadata.csv',
                                base_size = config.TRAIN.BASE_SIZE,
                                crop_size= config.TRAIN.IMAGE_SIZE,
                                multiscale = config.TRAIN.MULTI_SCALE,
                                multiscale_factor = config.TRAIN.MULTI_SCALE_FACTOR)
    
    val_dataset = BaseDataset(root = config.DATASET.ROOT,
                                list_path = 'metadata.csv',
                                stage = 'val',
                                num_samples = None,
                                base_size = config.TRAIN.BASE_SIZE,
                                crop_size= config.TRAIN.IMAGE_SIZE,
                                multiscale = config.TRAIN.MULTI_SCALE,
                                multiscale_factor = config.TRAIN.MULTI_SCALE_FACTOR)
                                
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        )

    model = UNet(n_channels = 3)
    model = DepthModel((480,640))

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD([{'params':
                                  filter(lambda p: p.requires_grad,
                                         model.parameters()),
                                  'lr': config.TRAIN.LR}],
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam([{'params':
                                  filter(lambda p: p.requires_grad,
                                         model.parameters()),
                                  'lr': config.TRAIN.LR}],
                                lr=config.TRAIN.LR)
    else:
        raise ValueError('Only Support SGD or ADAM optimizer')

    
    criterion = DepthLoss()
    model = FullModel(model, criterion)
    model = model.to(device)


    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    gpus = config.GPUS
    epoch_iters = np.int64(train_dataset.__len__() / 
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    best_checkpoints = []
    last_epoch = 0



    # for i_iter, batch in enumerate(trainloader):
    #     images, labels, _, _ = batch
    #     labels = torch.unsqueeze(labels, 1)
    #     losses, outputs = model(images,labels)
    #     break

    # for epoch in range(0,end_epoch):
    #     train(config, epoch, end_epoch,
    #               epoch_iters, 
    #               config.TRAIN.LR, num_iters,
    #               trainloader, optimizer, model, 
    #               writer_dict, device)

    for epoch in range(last_epoch, end_epoch):
        
        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict,
                  device)

        # validation
        valid_loss, valid_mse, valid_mae, valid_abs_rel, valid_delta1, valid_delta2, valid_delta3, valid_log10 = validate(config, 
                    valloader, model, writer_dict, device)
        

        if args.local_rank:
            continue

        # checkpointing
        model_state_dict = model.module.state_dict() if distributed else model.state_dict()

        if len(best_checkpoints) < config.SAVE_TOP_K or valid_delta1 > best_checkpoints[-1]["valid_delta1"]:
            checkpoint_path = os.path.join(checkpoints_dir, f'epoch_{epoch + 1}_loss_{valid_loss:0.3f}_delta1_{valid_delta1:0.3f}.pth')
            torch.save(model_state_dict, checkpoint_path)
            logger.info(f'=> saving top {config.SAVE_TOP_K} checkpoint to {checkpoint_path}')
            
            best_checkpoints.append({'valid_loss': valid_loss, 'valid_delta1':valid_delta1, 'path': checkpoint_path})
            best_checkpoints = sorted(best_checkpoints, key=itemgetter('valid_delta1'), reverse=True) # best model first
        else:
            logger.info(f'=> Checkpoint not in top delta1 {config.SAVE_TOP_K}.')

        if len(best_checkpoints) > config.SAVE_TOP_K:
            outdated_checkpoint = best_checkpoints.pop()
            os.remove(outdated_checkpoint["path"])

        logger.info('=> saving checkpoint to {}'.format(
            os.path.join(checkpoints_dir,'last.pth.tar')))
        torch.save({
            'epoch': epoch+1,
            'best_checkpoints': best_checkpoints,
            'state_dict': model_state_dict,
            'optimizer': optimizer.state_dict(),
            }, os.path.join(checkpoints_dir,'last.pth.tar'))

        if epoch == end_epoch - 1:
            torch.save(model_state_dict,
                os.path.join(checkpoints_dir, 'final_state.pth'))

            writer_dict['writer'].close()
            end = timeit.default_timer()
            logger.info('Hours: %d' % np.int((end-start)/3600))
            logger.info('Done')

        # logging
        logging.info(SEP)
        logging.info('Validation Results:')
        logging.info(SEP)

        msg = 'Loss: {:.3f}, Best_delta1: {: 4.4f}\nMSE: {: 4.4f}, MAE:{: 4.4f}, ABS REL:{: 4.4f} , delta1:{: 4.4f} , delta2:{: 4.4f} , delta3:{: 4.4f}, log10:{: 4.4f}'.format(
                valid_loss,  best_checkpoints[0]["valid_delta1"], valid_mse, valid_mae, valid_abs_rel, valid_delta1, valid_delta2, valid_delta3, valid_log10)
        logging.info(msg)


    

if __name__ == '__main__':
    main()