import logging
import os
import time

import numpy as np
import pandas as pd
import numpy.ma as ma
from tqdm import tqdm
import seaborn as sn

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

from utils.utils import adjust_learning_rate
from utils.utils import AverageMeter
from core.metrics import evaluate_depth_metrics




def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device):
    print("TRAINING")
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = torch.unsqueeze(labels, 1)
        labels = labels.to(device)
        losses, prediction1 = model(images, labels)

        loss = losses.mean()

        reduced_loss = loss 
        ###################
        # WILL ONLY WORK ON ONLY 1 GPU, otherwise => implement reduce tensor
        ##################

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(reduced_loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)
        
        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average()
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict, device):
    
    #if multigpu get world size and get rank
    print("VALIDATION")
    model.eval()
    ave_loss = AverageMeter()
    ave_mse = AverageMeter()
    ave_mae = AverageMeter()
    ave_abs_rel = AverageMeter()
    ave_delta1 = AverageMeter()
    ave_delta2 = AverageMeter()
    ave_delta3 = AverageMeter()

    with torch.no_grad():
        
        for _, batch in enumerate(testloader):
            images, labels, _, _ = batch
            size = labels.size()
            images = images.to(device)
            labels = torch.unsqueeze(labels, 1)
            labels = labels.to(device)

            losses, pred = model(images, labels)
            loss = losses.mean()
            mse, mae, abs_rel, delta1, delta2, delta3 = evaluate_depth_metrics(pred, labels)
            # reduce loss if multi gpu => TODO: implement reduce tensor, get world size 
            reduced_loss = loss
            ave_loss.update(reduced_loss.item())
            ave_mse.update(mse)
            ave_mae.update(mae)
            ave_abs_rel.update(abs_rel)
            ave_delta1.update(delta1)
            ave_delta2.update(delta2)
            ave_delta3.update(delta3)

    print_loss = ave_loss.average() # /world_size
    print_mse = ave_mse.average()
    print_mae = ave_mae.average()
    print_abs_rel = ave_abs_rel.average()
    print_delta1 = ave_delta1.average()
    print_delta2 = ave_delta1.average()
    print_delta3 = ave_delta3.average()
    # if rank == 0: 
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', print_loss, global_steps)
    writer.add_scalar('valid_mse', print_mse, global_steps)
    writer.add_scalar('valid_mae', print_mae, global_steps)
    writer.add_scalar('valid_abs_rel', print_abs_rel, global_steps)
    writer.add_scalar('valid_delta1', print_delta1, global_steps)
    writer.add_scalar('valid_delta2', print_delta2, global_steps)
    writer.add_scalar('valid_delta3', print_delta3, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, print_mse, print_mae, print_abs_rel, print_delta1, print_delta2, print_delta3