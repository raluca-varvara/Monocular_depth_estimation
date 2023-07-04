import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics.functional import image_gradients
from torchmetrics.functional import structural_similarity_index_measure
from core.vnl_loss import VNL

def depth_loss_function(y_true, y_pred, device, theta=0.2, maxDepthVal=1000.0/10.0):
  
  # Point-wise depth
  loss_depth = torch.mean(torch.abs(y_pred - y_true), axis=-1)

  # Edges
  dy_true, dx_true = image_gradients(y_true)
  dy_pred, dx_pred = image_gradients(y_pred)
  l_edges = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), axis=-1)

  # Structural similarity (SSIM) index
  l_ssim = torch.clip((1 - structural_similarity_index_measure(y_true, y_pred, data_range = maxDepthVal)) * 0.5, 0, 1) 
  
  # Virtual Normal Loss
  vnl_loss = VNL(y_true.shape, device) 

  # Weights
  w1 = 0.1
  w2 = 1.0
  w3 = 10.0
  w4 = 10.0
  # modified

  return  (w1 * torch.mean(loss_depth))  + (w2 * torch.mean(l_edges)) + (w3 * l_ssim)  + (w4 * vnl_loss(y_true,y_pred)), w1*torch.mean(loss_depth),  w2*torch.mean(l_edges), w3*l_ssim, w4*vnl_loss(y_true,y_pred)

class DepthLoss(nn.Module):
    def __init__(self, device):
        super(DepthLoss, self).__init__()    
        self.device = device        

    def forward(self, score, target):
        loss, l_depth, l_edges, l_ssim, l_vnl = depth_loss_function(target, score, self.device)

        return loss, l_depth, l_edges, l_ssim, l_vnl
