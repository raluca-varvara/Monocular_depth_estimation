
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
import logging
from pathlib import Path
import os
import glob

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
        
def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr
    
class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss
    def forward(self, inputs, labels):
        outputs, outputs_softmax = self.model(inputs)
        loss, l_depth, l_edges, l_ssim, l_vnl = self.loss(outputs, labels)
        return torch.unsqueeze(loss, 0), outputs, l_depth, l_edges, l_ssim, l_vnl

def create_logger(cfg, cfg_name, experiment_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / model / experiment_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

   
    versions = glob.glob(f"{final_output_dir}/v.*")
    if versions:
        last_version = max([int(version.split('.')[-1]) for version in versions])
        exp_version = last_version + 1
    else:
        exp_version = 1

    final_output_dir = final_output_dir / f"v.{exp_version}"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = f'{model}_{experiment_name}_v.{exp_version}.log'
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = final_output_dir / "log_dir"
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = final_output_dir / "checkpoints"
    print('=> creating {}'.format(checkpoints_dir))
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir), str(checkpoints_dir)