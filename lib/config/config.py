from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 1
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
# _C.PIN_MEMORY = True
# _C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'unet_monocular'
_C.MODEL.PRETRAINED = ''

_C.LOSS = CN()
_C.LOSS.TYPE = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'indoor_monocular'
_C.DATASET.METADATA = ''

# training
_C.TRAIN = CN()

_C.TRAIN.IMAGE_SIZE = [848, 480]  # width * height
_C.TRAIN.BASE_SIZE = 848
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.MULTI_SCALE_FACTOR = 1.5


_C.TRAIN.LR_FACTOR = 0.1
# _C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.01
# _C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 484

_C.TRAIN.RESUME = False

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
# only using some training samples
_C.TRAIN.NUM_SAMPLES = 0

# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [848, 480]  # width * height
_C.TEST.BASE_SIZE = 848

_C.TEST.BATCH_SIZE_PER_GPU = 32
# only testing some samples
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg,  dataset_file_path, training_file_path, model_file_path):
    cfg.defrost()
    
    # split the cofig file in 3 
    cfg.merge_from_file(dataset_file_path)
    cfg.merge_from_file(training_file_path)
    cfg.merge_from_file(model_file_path)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

