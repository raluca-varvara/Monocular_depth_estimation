import torch 
import math
from typing import Tuple

import torch
import numpy as np

SEP = "##########################################################"

def evaluate_depth_metrics(output, target):
    valid_mask = target>0
    output = output[valid_mask]
    target = target[valid_mask]
    # calc diff
    diff_matrix = torch.abs(output - target)

    # mse, mae
    mse = torch.mean(torch.pow(diff_matrix, 2),axis = -1) 
    mae = torch.mean(diff_matrix,axis=-1)

    # abs rel
    real_matrix = torch.div(diff_matrix, target)
    abs_rel = torch.mean(real_matrix, axis = -1) 
    # lg10
    lg10_matrix = torch.abs(torch.log10(output) - torch.log10(target))
    lg10 = torch.mean(lg10_matrix,axis=-1)

    # delta
    y_over_z = torch.div(output, target)
    z_over_y = torch.div(target, output)
    max_ratio = torch.maximum(y_over_z, z_over_y)
    delta1 = torch.mean(
        torch.le(max_ratio, 1.25).float(), axis = -1) 
    # delta1 = torch.le(max_ratio, 1.25).float()
    delta2 = torch.mean(
        torch.le(max_ratio, math.pow(1.25, 2)).float(), axis = -1) 
    delta3 = torch.mean(
        torch.le(max_ratio, math.pow(1.25, 3)).float(), axis = -1) 

    return float(mse.data.cpu().numpy()), float(mae.data.cpu().numpy()), float(abs_rel.data.cpu().numpy()), float(delta1.data.cpu().numpy()), float(delta2.data.cpu().numpy()), float(delta3.data.cpu().numpy())
    

if __name__ == '__main__':
    output = torch.rand(4,1,128,128)
    # target = output
    target = torch.rand(4,1,128,128)
    print(evaluate_depth_metrics(output, target))
