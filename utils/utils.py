import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
from yacs.config import CfgNode


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    log_file = str(final_log_file)
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    print('=> logging to {}'.format(final_log_file))

    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)

    if not os.path.exists(cfg.TRAIN.LOG_DIR):
        os.makedirs(cfg.TRAIN.LOG_DIR)

    return logger, str(final_output_dir)


def get_optimizer(cfg, model):
    optimizer = None
    cfg_opt = str(cfg.TRAIN.OPTIMIZER).lower()
    if cfg_opt == 'sgd':
        optimizer = optim.SGD(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg_opt == 'adamw':
        optimizer = optim.AdamW(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD,
            betas=(cfg.TRAIN.BETAS[0], cfg.TRAIN.BETAS[1])
        )
    elif cfg_opt == 'adam':
        optimizer = optim.Adam(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD,
            betas=(cfg.TRAIN.BETAS[0], cfg.TRAIN.BETAS[1])
        )
    elif cfg_opt == 'rmsprop':
        optimizer = optim.RMSprop(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED,
            eps=cfg.TRAIN.EPS
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'best_model.pth'))


def getFloat32Info():
    import torch

    info = torch.finfo(torch.float32)
    print(f"最大值 (max): {info.max}")  # 3.4028234663852886e+38
    print(f"最小正正规数 (tiny): {info.tiny}")  # 1.1754943508222875e-38
    print(f"机器精度 (eps): {info.eps}")  # 1.1920928955078125e-07
    return info.max, info.tiny, info.eps


def cfg_to_dict(cfg_node, key_list=None):
    """递归将 CfgNode 转换为 dict"""
    if key_list is None:
        key_list = []
    if not isinstance(cfg_node, CfgNode):
        return cfg_node
    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
        cfg_dict[k] = cfg_to_dict(v, key_list + [k])
    return cfg_dict
