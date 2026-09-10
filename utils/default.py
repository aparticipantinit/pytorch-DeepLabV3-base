import os

from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 0
_C.PRINT_FREQ = 20
_C.PRINT_INFO = True
_C.PRINT_LOG_INFO = False
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'my_model'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [512, 512]  # width * height, ex: 512 * 512
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = 'dataset/'
_C.DATASET.DATASET = 'crop_image'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'tif'

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.0003

_C.TRAIN.OPTIMIZER = 'adamW'
_C.TRAIN.LOG_DIR = 'adamW'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.00001
_C.TRAIN.NESTEROV = True
_C.TRAIN.RMSPROP_CENTERED = True
_C.TRAIN.BETAS = [0.9, 0.999]
_C.TRAIN.EPS = 0.00001

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.SAVE_PER_EPOCH = 1
_C.TRAIN.EARLY_STOP = 20
_C.TRAIN.SAVE_LOSS_PER_EPOCH = 1

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.CLEAR_CUDA_PER_EPOCH = 2  # 每2个周期清理一次cuda显存，防止显存被垃圾占满而溢出崩溃

_C.TRAIN.RANDOM_CROP = True
_C.TRAIN.RANDOM_HORIZONTAL_FLIP = True
_C.TRAIN.RANDOM_VERTICAL_FLIP = True
_C.TRAIN.RANDOM_ROTATION = True
_C.TRAIN.ROTATION_RANGE = 30
_C.TRAIN.RANDOM_AFFINE = False
_C.TRAIN.RANDOM_AUTO_CONTRAST = False

_C.TRAIN.LOG_LOSS_PRE_EPOCH = 1
_C.TRAIN.VALID_WITH_IOU = False

_C.TRAIN.STEP_BY_SINGLE_EPOCH = False

_C.TRAIN.USE_PARALLELED = True
# 若此设备上的cuda可用，且可用的cuda设备数量大于1，则使用torch.nn.DataParallel将模型和数据分发到多个 GPU 上并行计算

_C.VALID = CN()
_C.VALID.DATASET = 'crop_image'
_C.VALID.DATA_FORMAT = 'tif'
_C.VALID.ROOT = 'dataset/'
_C.VALID.TEST_SET = 'test'

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.SHIFT_HEATMAP = False
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False


def get_cfg_defaults():
    return _C.clone()


__all__ = ['get_cfg_defaults']
