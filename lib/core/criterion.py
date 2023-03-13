import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics.functional import image_gradients
from torchmetrics.functional import structural_similarity_index_measure

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
  
  # Point-wise depth
  l_depth = torch.mean(torch.abs(y_pred - y_true), axis=-1)

  # Edges
  dy_true, dx_true = image_gradients(y_true)
  dy_pred, dx_pred = image_gradients(y_pred)
  l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), axis=-1)

  # Structural similarity (SSIM) index
  l_ssim = torch.clip((1 - structural_similarity_index_measure(y_true, y_pred, data_range = maxDepthVal)) * 0.5, 0, 1) 

  # Weights
  w1 = 1.0
  w2 = 1.0
  w3 = theta
  return  (w3 * torch.mean(l_depth)) + (w1 * l_ssim) + (w2 * torch.mean(l_edges))  

def depth_acc(y_true, y_pred):
  return 1.0 - depth_loss_function(y_true, y_pred)


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()        

    def forward(self, score, target):
        loss = depth_loss_function(target, score)

        return loss
