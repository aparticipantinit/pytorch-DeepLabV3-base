import os
import pprint
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from model import MyModelUtil

from utils import create_logger, cfg_to_dict
from config import args_c


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)
add_path(this_dir)


def main():
    args = args_c('./my_model.yaml')
    config = args.getConfig()
    logger, final_output_dir = create_logger(
        config, args.cfg, 'train')

    log_config = cfg_to_dict(config)
    logger.info(pprint.pformat(log_config, width=50))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = MyModelUtil(num_classes=2)
    model.set_final_output_dir(final_output_dir)
    model.set_config(config)

    # Data loading code
    train_dir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    val_dir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)
    model.load_dataset(train_dir, val_dir, config.MODEL.IMAGE_SIZE)

    model.set_max_epoch(config.TRAIN.END_EPOCH)
    model._try_assert()
    model.start_train()


if __name__ == '__main__':
    main()
