import logging
import os
import time

import cv2
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def train(config, epoch, num_epoch, epoch_iters, base_lr, num_iters,
         trainloader, optimizer, model, writer_dict, device):
    print("TRAINING")
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    ave_loss_depth = AverageMeter()
    ave_loss_edges = AverageMeter()
    ave_loss_ssim =  AverageMeter()
    ave_loss_vnl = AverageMeter()
    tic = time.time()
    cur_iters = epoch * epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter, batch in enumerate(trainloader):
        images, labels, _, _ = batch
        images = images.to(device)
        labels = torch.unsqueeze(labels, 1)
        labels = labels.to(device)
        losses, prediction1, l_depth, l_edges, l_ssim, l_vnl = model(images, labels)

        loss = losses.mean()
        loss_depth = l_depth.mean()
        loss_ssim = l_ssim.mean()
        loss_edges = l_edges.mean()
        loss_vnl = l_vnl.mean()

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
        ave_loss_depth.update(loss_depth.item())
        ave_loss_edges.update(loss_edges.item())
        ave_loss_ssim.update(loss_ssim.item())
        ave_loss_vnl.update(loss_vnl.item())

        # lr = adjust_learning_rate(optimizer,
        #                           base_lr,
        #                           num_iters,
        #                           i_iter+cur_iters)
        lr = get_lr(optimizer)
        if i_iter % config.PRINT_FREQ == 0:
            print_loss = ave_loss.average()
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, print_loss)
            logging.info(msg)
            
            
            writer.add_scalar('train_loss', print_loss, global_steps)
            writer.add_scalar('train_loss_depth' , ave_loss_depth.average(), global_steps)
            writer.add_scalar('train_loss_edges' , ave_loss_edges.average(), global_steps)
            writer.add_scalar('train_loss_ssim' , ave_loss_ssim.average(), global_steps)
            writer.add_scalar('train_loss_vnl' , ave_loss_vnl.average(), global_steps)
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
    ave_log10 = AverageMeter()

    with torch.no_grad():
        
        for _, batch in enumerate(testloader):
            images, labels, _, _ = batch
            size = labels.size()
            images = images.to(device)
            labels = torch.unsqueeze(labels, 1)
            labels = labels.to(device)

            losses, pred, l_depth, l_edges, l_ssim, l_vnl = model(images, labels)
            loss = losses.mean()
            mse, mae, abs_rel, delta1, delta2, delta3, log10 = evaluate_depth_metrics(pred, labels)
            # reduce loss if multi gpu => TODO: implement reduce tensor, get world size 
            reduced_loss = loss
            ave_loss.update(reduced_loss.item())
            ave_mse.update(mse)
            ave_mae.update(mae)
            ave_abs_rel.update(abs_rel)
            ave_delta1.update(delta1)
            ave_delta2.update(delta2)
            ave_delta3.update(delta3)
            ave_log10.update(log10)

    print_loss = ave_loss.average() # /world_size
    print_mse = ave_mse.average()
    print_mae = ave_mae.average()
    print_abs_rel = ave_abs_rel.average()
    print_delta1 = ave_delta1.average()
    print_delta2 = ave_delta1.average()
    print_delta3 = ave_delta3.average()
    print_log10 = ave_log10.average()
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
    writer.add_scalar('valid_log10', print_log10, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, print_mse, print_mae, print_abs_rel, print_delta1, print_delta2, print_delta3, print_log10

def test(config, testloader, model, sv_dir=''):

    model.eval()

    ave_loss = AverageMeter()
    ave_mse = AverageMeter()
    ave_mae = AverageMeter()
    ave_abs_rel = AverageMeter()
    ave_delta1 = AverageMeter()
    ave_delta2 = AverageMeter()
    ave_delta3 = AverageMeter()
    ave_log10 = AverageMeter()

    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, label, size, name, image_true = batch

            label = torch.unsqueeze(label, 1)
            size = size[0]

            loss, pred, l_depth, l_edges, l_ssim, l_vnl = model(image, label)
            mse, mae, abs_rel, delta1, delta2, delta3, log10 = evaluate_depth_metrics(pred, label)
            # reduce loss if multi gpu => TODO: implement reduce tensor, get world size 

            reduced_loss = loss
            ave_loss.update(reduced_loss.item())
            ave_mse.update(mse)
            ave_mae.update(mae)
            ave_abs_rel.update(abs_rel)
            ave_delta1.update(delta1)
            ave_delta2.update(delta2)
            ave_delta3.update(delta3)
            ave_log10.update(log10)

            pred = torch.squeeze(pred)
            print("#####################################################")
            print("#####################################################")
            tensor = pred*255
            print(tensor.type(torch.int8))
            
            print("#####################################################")
            print("#####################################################")


            sv_path = str(sv_dir / "predictions") + '/' + name[0] + ".png"
            tensor  = tensor.cpu().numpy() # make sure tensor is on cpu
            cv2.imwrite(sv_path,tensor)

            sv_path = str(sv_dir / "images") + '/' + name[0] + ".png"
            image_true = torch.squeeze(image_true)
            image_true  = image_true.cpu().numpy()
            cv2.imwrite(sv_path, image_true)

            # test_dataset.save_pred(pred, sv_path, name)

        print_loss = ave_loss.average() 
        print_mse = ave_mse.average()
        print_mae = ave_mae.average()
        print_abs_rel = ave_abs_rel.average()
        print_delta1 = ave_delta1.average()
        print_delta2 = ave_delta1.average()
        print_delta3 = ave_delta3.average()
        print_log10 = ave_log10.average()

        return print_loss, print_mse, print_mae, print_abs_rel, print_delta1, print_delta2, print_delta3, print_log10