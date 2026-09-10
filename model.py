import logging
import os
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
from torchvision.models import ResNet101_Weights
from torchvision.transforms import v2

from MyDataset import MyBuildingDataset
from utils.lost import CombinedLoss

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from utils import pixel_accuracy, save_checkpoint, get_optimizer, accuracy_and_iou

from tqdm import tqdm

logger = logging.getLogger(__name__)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride):
        super(ASPP, self).__init__()
        rates = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
        # 1x1 卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 3x3 空洞卷积，空洞率由 rates 指定
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=rates[0], dilation=rates[0])
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=rates[1], dilation=rates[1])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   padding=rates[2], dilation=rates[2])
        # 全局平均池化分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_avg = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 关键修正：拼接后降维到 out_channels
        self.conv_out = nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3_1(x)
        out3 = self.conv3x3_2(x)
        out4 = self.conv3x3_3(x)
        # 全局池化分支
        out5 = self.avg_pool(x)
        out5 = self.conv_avg(out5)
        out5 = F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)

        # 拼接所有分支
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        # 降维到 out_channels
        out = self.conv_out(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=2, output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        # 使用 ResNet101 作为骨干网络
        backbone = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 输出通道 256，尺寸 1/4
        self.layer2 = backbone.layer2  # 输出通道 512，尺寸 1/8
        self.layer3 = backbone.layer3  # 输出通道 1024，尺寸 1/16
        self.layer4 = backbone.layer4  # 输出通道 2048，尺寸 1/32

        # ASPP 模块，输入通道为 layer4 的输出 2048，输出通道设为 256
        self.aspp = ASPP(in_channels=2048, out_channels=256, output_stride=output_stride)

        # 解码器：降低浅层特征通道数（layer1 输出 256 -> 48）
        # 用于处理低层特征，将其通道数降至与ASPP输出一致 (256)
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),  # 256是ResNet layer1的输出通道数
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        # 融合后的特征处理
        self.decoder_fusion = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # 最终分类层
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # ---- 编码器 ----
        x0 = self.layer0(x)  # 1/2
        x1 = self.layer1(x0)  # 1/4，保留用于跳跃连接
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        # ---- ASPP ----
        aspp_out = self.aspp(x4)  # 输出通道 256，尺寸 1/32

        # ---- 解码器 ----
        # 1. 上采样 ASPP 输出至 1/4 尺寸
        decoder_out = F.interpolate(aspp_out, size=x1.size()[2:], mode='bilinear', align_corners=True)
        # 2. 处理浅层特征
        x1_reduced = self.decoder_conv(x1)  # 256 -> 48
        # 3. 拼接
        concat = torch.cat([decoder_out, x1_reduced], dim=1)
        # 4. 最终卷积
        final_out = self.decoder_fusion(concat)
        # 5. 最终分类，并上采样回原始输入尺寸
        final_out = self.classifier(final_out)
        final_out = F.interpolate(final_out, size=input_size, mode='bilinear', align_corners=True)

        return final_out


# noinspection PyTypeChecker
class MyModelUtil:
    def __init__(self, num_classes=2, output_stride=16):
        self.best_loss_epoch = 0
        self.early_stop = True
        self.early_stop_number = 0
        self.gpus = [0, ]
        self.val_dir = None
        self.train_dir = None
        self.best_perf = 0
        self.image_size = None
        self.is_best_model = False
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.origin_model = DeepLabV3Plus(num_classes=num_classes, output_stride=self.output_stride)
        self.origin_model.to(self.device)
        self.model = self.origin_model
        self.train_dataLoader = None
        self.val_dataLoader = None
        self.max_epoch = 100
        self.optimizer = torch.optim.Adam(self.origin_model.parameters(), lr=0.001)
        self.criterion = None
        self.last_epoch = 0
        self.epoch = 0
        self.log_epoch = self.epoch + 1
        self.best_acc = 0.0
        self.total_train_time_used = 0
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])  # ImageNet数据集的标准化均值和标准差参数
        self.train_transform_list = []
        self.scheduler = None
        self.config = None
        self.final_output_dir = "./final_output"

        self.train_loss_list = []
        self.val_loss_list = []
        self.train_accuracy_list = []
        self.val_accuracy_list = []

        self.__initialize = True
        self.__init_train = False
        self.__headers_train = [
            "Training Epoch", "Time(s)", "AvgTime(s)", "Speed(samples/s)",
            "Avg Loss", "Avg Accuracy"
        ]
        self.__headers_valid = [
            "Validating Epoch", "Total Time(s)", "AvgTime(s)",
            "Avg Loss", "Error@accuracy", "Avg Accuracy"
        ]
        if torch.cuda.is_available():
            logger.info(f"cuda 可用设备数: {torch.cuda.device_count()}")
            logger.info(f"cuda 设备名称: {torch.cuda.get_device_name(self.device)}")
            device_capability = torch.cuda.get_device_capability(self.device)
            logger.info(f"cuda 设备的计算能力: {device_capability[0]}.{device_capability[1]}")
            logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
            logger.info(f"GPU Memory has used: {torch.cuda.device_memory_used() / 1024 ** 3:.2f} GB")
            torch.cuda.empty_cache()
        else:
            logger.info("cuda 不可用，使用cpu计算")
            logger.info(f"cpu 可用设备数: {torch.cpu.device_count()}")

    def check_point(self, checkpoint_path=None):
        if self.config.TRAIN.RESUME:
            if checkpoint_path is None:
                model_state_file = os.path.join(self.final_output_dir,
                                                'checkpoint.pth')
            else:
                model_state_file = checkpoint_path
            if os.path.isfile(model_state_file):
                checkpoint = torch.load(model_state_file)
                self.last_epoch = checkpoint['epoch']
                self.best_perf = checkpoint['best_accuracy']
                self.origin_model.load_state_dict(checkpoint['state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.last_epoch = checkpoint['epoch']
                self.early_stop_number = checkpoint['early_stop_number']
                self.best_loss_epoch = checkpoint['best_loss_epoch']
                self.train_loss_list = checkpoint['train_loss_list']
                self.val_loss_list = checkpoint['val_loss_list']
                self.train_accuracy_list = checkpoint['train_accuracy_list']
                self.val_accuracy_list = checkpoint['val_accuracy_list']
                self.is_best_model = checkpoint['is_best']
                self.total_train_time_used = checkpoint['total_train_time_used']
                text = (
                    "=> loaded checkpoint (epoch {}), and the best epoch is at (epoch {}) with the (accuracy {:.4f}%)".
                    format(self.last_epoch, self.best_loss_epoch + 1, self.best_perf))
                print("检查点已加载,epoch:", self.last_epoch)
                logger.info(text)
            else:
                text = "=> no checkpoint file found at '{}'".format(model_state_file)
                logger.info(text)
                print("在给定的路径下 (", model_state_file, ") 没找到检查点文件")

    def set_final_output_dir(self, final_output_dir):
        self.final_output_dir = final_output_dir

    def set_config(self, config):
        self.config = config
        self.last_epoch = self.config.TRAIN.BEGIN_EPOCH
        self.max_epoch = self.config.TRAIN.END_EPOCH
        self.gpus = list(self.config.GPUS)

        self.early_stop = self.config.TRAIN.EARLY_STOP  # 早停

        # define loss function (criterion) and optimizer
        self.criterion = CombinedLoss(ce_weight=1.0, dice_weight=2.0, dice_smooth=1e-9,
                                      cross_entropy_weight=torch.tensor([1.0, 3.0]).to(self.device))
        self.optimizer = get_optimizer(config, self.origin_model)

        if isinstance(config.TRAIN.LR_STEP, list):
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                self.last_epoch - 1
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                self.last_epoch - 1
            )
        if self.config.TRAIN.VALID_WITH_IOU:
            self.__headers_valid = [
                "Validating Epoch", "Total Time(s)", "AvgTime(s)",
                "Avg Loss", "Error@accuracy", "Avg Accuracy", "Ayg IOU"
            ]
        self.check_point()

    def load_model(self, model_path):
        self.origin_model.load_state_dict(torch.load(model_path))

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.optimizer.zero_grad()

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_last_epoch(self, epoch):
        self.last_epoch = epoch

    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def set_data_loaders(self, train_data_loader, val_data_loader):
        self.train_dataLoader = train_data_loader
        self.val_dataLoader = val_data_loader

    def load_dataset(self, train_dir, val_dir, image_size):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.image_size = image_size
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        elif isinstance(image_size, (list, tuple)):
            self.image_size = (image_size[0], image_size[1])
        elif isinstance(image_size, (float, str)):
            image_size = int(image_size)
            self.image_size = (image_size, image_size)
        if train_dir is not None:
            self.train_transform_list = [v2.Resize(image_size)]
            if self.config.TRAIN.RANDOM_CROP:
                self.train_transform_list.append(
                    v2.RandomCrop(self.image_size, pad_if_needed=True, fill=0))
            if self.config.TRAIN.RANDOM_HORIZONTAL_FLIP:
                self.train_transform_list.append(v2.RandomHorizontalFlip())
            if self.config.TRAIN.RANDOM_VERTICAL_FLIP:
                self.train_transform_list.append(v2.RandomVerticalFlip())
            if self.config.TRAIN.RANDOM_ROTATION:
                self.train_transform_list.append(
                    v2.RandomRotation(self.config.TRAIN.ROTATION_RANGE, fill=0))
            if self.config.TRAIN.RANDOM_AUTO_CONTRAST:
                self.train_transform_list.append(
                    v2.RandomAutocontrast()
                )
            if self.config.TRAIN.RANDOM_AFFINE:
                self.train_transform_list.append(
                    v2.RandomAffine(self.config.TRAIN.ROTATION_RANGE, fill=0)
                )
            train_dataset = MyBuildingDataset(
                train_dir,
                transform=v2.Compose(self.train_transform_list),
                normalize=self.normalize,
            )
            self.train_dataLoader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.TRAIN.BATCH_SIZE_PER_GPU * len(self.gpus),
                shuffle=True,
                num_workers=self.config.WORKERS,
                pin_memory=True
            )
            print(f"从 {train_dir} 加载了 {len(train_dataset)} 个训练集数据用于训练")
        if val_dir is not None:
            valid_dataset = MyBuildingDataset(val_dir, transform=v2.Compose([
                v2.ToTensor(),
                v2.Resize(self.image_size),
            ]), normalize=self.normalize)

            self.val_dataLoader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.config.TEST.BATCH_SIZE_PER_GPU * len(self.gpus),
                shuffle=False,
                num_workers=self.config.WORKERS,
                pin_memory=True
            )
            print(f"从 {val_dir} 加载了 {len(valid_dataset)} 个验证集数据用于每轮训练后的验证")

    def check(self):
        self.__init_train = (self.optimizer is not None
                             and self.criterion is not None
                             and self.last_epoch < self.max_epoch
                             and self.train_dataLoader is not None
                             and self.val_dataLoader is not None
                             and self.scheduler is not None
                             and self.__initialize)
        return self.__init_train

    def _try_assert(self):
        assert self.__initialize
        assert (self.optimizer is not None)
        assert (self.criterion is not None)
        assert (self.last_epoch < self.max_epoch)
        assert (self.train_dataLoader is not None)
        assert (self.val_dataLoader is not None)
        assert (self.scheduler is not None)

    def train_one_epoch(self):
        """train for one epoch"""
        if not (self.__initialize or self.__init_train):
            raise RuntimeError("未完成训练前的初始化，相关数据未完成准备")
        if self.epoch > self.max_epoch:
            return False
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()

        # switch to train mode
        self.model.train()
        end = time.time()

        pbar = tqdm(enumerate(self.train_dataLoader),
                    total=len(self.train_dataLoader),
                    desc=f"Epoch: ({self.log_epoch}/{self.max_epoch}) Training ")
        pbar.set_postfix({'loss': '?.??', 'accuracy': '?.??%'})
        input_size = 0
        for i, (_input, target) in pbar:
            # measure data loading time
            data_time.update(time.time() - end)
            self.optimizer.zero_grad()
            input_size = _input.size(0)
            _input = _input.to(self.device)
            # compute output
            output = self.model(_input)
            target = target.to(self.device)

            loss = self.criterion(output, target)

            # compute gradient and do update step
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), _input.size(0))

            acc = pixel_accuracy(output, target)
            accuracy.update(acc, _input.size(0))

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{acc:.4f}%',
            })

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        return batch_time, input_size, data_time, losses, accuracy

    def start_train(self):
        if not self.check():
            raise RuntimeError("初始化未完成")
        if self.config.TRAIN.USE_PARALLELED and torch.cuda.device_count() > 1 and self.device.type == 'cuda':
            self.model = torch.nn.DataParallel(self.origin_model, device_ids=self.gpus)
        else:
            self.model = self.origin_model
        self.model.to(self.device)
        torch.cuda.empty_cache()

        print(f"即将开始训练，从第 {self.last_epoch + 1} 轮开始,总共最多训练 {self.max_epoch} 轮")
        for epoch in range(self.last_epoch, self.max_epoch):
            self.epoch = epoch
            self.log_epoch = self.epoch + 1
            current_train_time_start = time.time()
            if self.config.TRAIN.STEP_BY_SINGLE_EPOCH:
                self.scheduler.step(self.epoch)
            batch_time, input_size, data_time, losses_train, accuracy_train = self.train_one_epoch()
            if self.config.TRAIN.STEP_BY_SINGLE_EPOCH:
                self.scheduler.step(self.log_epoch)
            else:
                self.scheduler.step()
            self.optimizer.zero_grad()
            if (self.config is not None) and self.config.PRINT_INFO and (epoch % self.config.PRINT_FREQ == 0):
                if self.config.PRINT_LOG_INFO:
                    speed = input_size / batch_time.val
                    msg = f'Training Epoch: [{self.log_epoch}/{self.max_epoch}]\t' \
                          f'Total Time {batch_time.sum:.3f}s \t' \
                          f'Avg Time ({batch_time.avg:.3f}s)\t' \
                          f'Speed {speed:.1f} samples/s\t' \
                          f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          f'Loss {losses_train.val:.5f} ({losses_train.avg:.5f})\t' \
                          f'Accuracy {accuracy_train.val:.3f}% ({accuracy_train.avg:.3f}%)\t'
                    logger.info(msg)
                row = [
                    self.log_epoch,  # Epoch
                    f"{batch_time.sum:.3f}",  # Time(s) 当前批次所用时间
                    f"{batch_time.avg:.3f}",  # AvgTime(s)
                    f"{input_size / batch_time.val:.1f}",  # Speed (samples/s)
                    f"{losses_train.avg:.5f}",  # AvgLoss
                    f"{accuracy_train.avg:.3f}%"  # AvgAcc
                ]
                print(tabulate([row, ], headers=self.__headers_train, tablefmt="simple", showindex=False))
            if self.config.TRAIN.VALID_WITH_IOU:
                batch_time, losses, accuracy, iou, _, _ = self.validate_with_IOU()
                perf_indicator = accuracy.avg
                if self.config.PRINT_INFO:
                    if self.config.PRINT_LOG_INFO:
                        msg = 'Test: Time {batch_time.sum:.3f}\t' \
                              'Loss {loss.avg:.4f}\t' \
                              'IOU {iou.avg:.4f}\t' \
                              'Error@accuracy {error1:.3f}%\t' \
                              'Accuracy {acc.avg:.3f}%\t'.format(
                            batch_time=batch_time, loss=losses, acc=accuracy,
                            error1=100 - accuracy.avg, iou=iou)
                        logger.info(msg)
                    row = [
                        self.log_epoch,
                        f'{batch_time.sum:.3f}',
                        f'{batch_time.avg:.3f}',
                        f'{losses.avg:.4f}',
                        f'{(100.0 - accuracy.avg):.3f}%',
                        f'{accuracy.avg:.3f}%',
                        f'{iou.avg:.3f}%'
                    ]
                    print(tabulate([row, ], headers=self.__headers_valid, tablefmt="simple", showindex=False))
            else:
                batch_time, losses, accuracy = self.validate()
                perf_indicator = accuracy.avg
                if self.config.PRINT_INFO:
                    if self.config.PRINT_LOG_INFO:
                        msg = 'Test: Time {batch_time.sum:.3f}\t' \
                              'Loss {loss.avg:.4f}\t' \
                              'Error@accuracy {error1:.3f}%\t' \
                              'Accuracy {acc.avg:.3f}%\t'.format(
                            batch_time=batch_time, loss=losses, acc=accuracy,
                            error1=100 - accuracy.avg)
                        logger.info(msg)
                    row = [
                        self.log_epoch,
                        f'{batch_time.sum:.3f}',
                        f'{batch_time.avg:.3f}',
                        f'{losses.avg:.4f}',
                        f'{(100.0 - accuracy.avg):.3f}%',
                        f'{accuracy.avg:.3f}%'
                    ]
                    print(tabulate([row, ], headers=self.__headers_valid, tablefmt="simple", showindex=False))
            if perf_indicator > self.best_perf:
                logger.info(
                    "Current epoch (epoch {}) is the best epoch with the (accuracy {:.4f}%) better than (accuracy {:.4f}%)".
                    format(self.log_epoch, perf_indicator, self.best_perf))
                self.best_perf = perf_indicator
                self.is_best_model = True
                self.best_loss_epoch = epoch
                self.early_stop_number = 0
            else:
                self.is_best_model = False
                self.early_stop_number += 1

            if self.log_epoch % self.config.TRAIN.LOG_LOSS_PRE_EPOCH == 0:
                self.train_loss_list.append((self.log_epoch, losses_train.avg))
                self.val_loss_list.append((self.log_epoch, losses.avg))
                self.train_accuracy_list.append((self.log_epoch, accuracy_train.avg))
                self.val_accuracy_list.append((self.log_epoch, accuracy.avg))
                if not os.path.exists(self.config.TRAIN.LOG_DIR):
                    os.makedirs(self.config.TRAIN.LOG_DIR)
                with open(os.path.join(self.config.TRAIN.LOG_DIR, 'train_loss.txt'), 'a') as f:
                    f.write(f"[Epoch {self.log_epoch}] ==> train loss={losses_train.avg}; "
                            f"train accuracy={accuracy_train.avg}%; "
                            f"valid loss={losses.avg}; "
                            f"valid accuracy={accuracy.avg}%; \n"
                            )

            logger.info('=> saving checkpoint to {}'.format(self.final_output_dir))
            current_train_time_end = time.time()
            current_used_time = current_train_time_end - current_train_time_start
            self.total_train_time_used += current_used_time
            save_checkpoint({
                'epoch': epoch + 1,
                'early_stop_number': self.early_stop_number,
                'model': self.config.MODEL.NAME,
                'state_dict': self.origin_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'perf': perf_indicator,
                'loss': losses_train.avg,
                'accuracy': accuracy_train.avg,
                'best_loss_epoch': int(self.best_loss_epoch),
                'train_loss_list': self.train_loss_list,
                'val_loss_list': self.val_loss_list,
                'train_accuracy_list': self.train_accuracy_list,
                'val_accuracy_list': self.val_accuracy_list,
                'is_best': self.is_best_model,
                'best_accuracy': self.best_perf,
                'total_train_time_used': self.total_train_time_used,
            }, self.is_best_model, self.final_output_dir, filename='checkpoint.pth')

            if (self.config.TRAIN.SAVE_PER_EPOCH == 0) or (self.log_epoch % self.config.TRAIN.SAVE_PER_EPOCH == 0):
                self.save_model(True)
            if (self.config.TRAIN.SAVE_LOSS_PER_EPOCH == 0) or (
                    self.log_epoch % self.config.TRAIN.SAVE_LOSS_PER_EPOCH) == 0:
                self.save_model_loss()

            logger.info('=> saving checkpoint to {} is success.'.format(self.final_output_dir))
            logger.info(
                f"current epoch use time {current_used_time:.4f} s , total used time {self.total_train_time_used:.4f} s.")

            if (self.config.TRAIN.CLEAR_CUDA_PER_EPOCH == 0) or (
                    self.log_epoch % self.config.TRAIN.CLEAR_CUDA_PER_EPOCH == 0):
                torch.cuda.empty_cache()

            if self.early_stop_number > self.early_stop:
                text = f'=> 在训练了 {self.early_stop_number} 轮后,模型依旧没有任何改进，依据配置结束训练'
                logger.info(text)
                print(text)
                break
        text = f"最好的一次训练结果出现在第{self.best_loss_epoch}轮"
        logger.info(text)
        print(text)
        self.save_model(save_all=True)
        self.save_model_loss()

    def save_model_loss(self):
        # 保存每轮损失值和精度的变化的折线图
        train_loss_data = np.array(self.train_loss_list)
        val_loss_data = np.array(self.val_loss_list)
        train_accuracy_data = np.array(self.train_accuracy_list)
        val_accuracy_data = np.array(self.val_accuracy_list)
        titles = [f'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
        y_labels = [f'train_loss', 'train_accuracy(%)', 'val_loss', 'val_accuracy(%)']
        datas = [train_loss_data, train_accuracy_data, val_loss_data, val_accuracy_data]
        for idx, (title, y_label) in enumerate(zip(titles, y_labels)):
            x = datas[idx][:, 0]
            y = datas[idx][:, 1]
            # 清除当前图形内容
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(x, y, marker='o', label='数据点')  # marker 参数用于标记数据点
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(title)
            ax.set_xlabel("epochs")
            ax.set_ylabel(y_label)
            ax.legend()
            ax.grid(True)  # 添加网格线
            fig.savefig(os.path.join(self.final_output_dir, title + '.jpg'), format="jpg", dpi=300)

    def save_model(self, save_all=False):
        final_model_state_file = os.path.join(self.final_output_dir,
                                              'final_state.pth')
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(self.origin_model.state_dict(), final_model_state_file)
        if save_all:
            final_model_file = os.path.join(self.final_output_dir,
                                            'model.pth')
            # 保存整个模型
            torch.save(self.origin_model, final_model_file)

    def validate(self):
        """evaluate on validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            pbar = tqdm(enumerate(self.val_dataLoader),
                        total=len(self.val_dataLoader),
                        desc=f"Epoch:({self.log_epoch}/{self.max_epoch}) Validation ")
            pbar.set_postfix({'loss': '?.??', 'accuracy': '?.??%'})
            for i, (_input, target) in pbar:
                # compute output
                _input = _input.to(self.device)
                output = self.model(_input)

                target = target.to(self.device)

                loss = self.criterion(output, target)
                # measure accuracy and record loss
                losses.update(loss.item(), _input.size(0))

                acc = pixel_accuracy(output, target)
                accuracy.update(acc, _input.size(0))
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'accuracy': f'{acc:.4f}%'})

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        return batch_time, losses, accuracy

    def validate_with_IOU(self):
        """evaluate on validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        accuracy = AverageMeter()
        iou = AverageMeter()
        iou_0 = AverageMeter()
        iou_1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            pbar = tqdm(enumerate(self.val_dataLoader),
                        total=len(self.val_dataLoader),
                        desc=f"Epoch:({self.log_epoch}/{self.max_epoch}) Validation ")
            pbar.set_postfix({'loss': '?.??', 'accuracy': '?.??%', 'IOU': '?.??%'})
            for i, (_input, target) in pbar:
                # compute output
                _input = _input.to(self.device)
                output = self.model(_input)

                target = target.to(self.device)

                input_size = _input.size(0)

                loss = self.criterion(output, target)
                # measure accuracy and record loss
                losses.update(loss.item(), input_size)

                acc, _mean_iou, iou_list = accuracy_and_iou(output, target, 2)

                accuracy.update(acc, input_size)
                iou.update(_mean_iou, input_size)
                iou_0.update(iou_list[0], input_size)
                iou_1.update(iou_list[1], input_size)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'accuracy': f'{acc:.4f}%',
                                  'IOU': f'{_mean_iou:.4f}%({iou_list[0]:.4f}%-{iou_list[1]:.4f}%)'})

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        return batch_time, losses, accuracy, iou, iou_0, iou_1

    def predict_file(self, filepath):
        """predict file"""
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image_size = image.shape[:2]
        # 通常包括：调整大小、转为张量、归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            self.normalize,
        ])
        image_tensor = transform(image).unsqueeze(0)  # 增加batch维度: (C, H, W) -> (1, C, H, W)

        self.model.eval()
        # --- 4. 模型预测 ---
        with torch.no_grad():  # 推理时关闭梯度计算
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)  # 输出形状: (1, 1, H, W) 或 (1, num_classes, H, W)

        # 概率图转为二值图
        pred_mask = torch.argmax(output, dim=1, keepdim=True).float()
        mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
        # 将二值图（0和1）映射到0和255，并转为uint8类型，便于保存为图像
        mask = (mask * 255).astype(np.uint8)
        return mask

    def get_model__name_parameters_namelist(self):
        for name, param in self.origin_model.named_parameters():
            yield name, param.shape, param.numel()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
