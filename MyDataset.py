import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors as TV
import warnings

warnings.filterwarnings('ignore')


class MyBuildingDataset(Dataset):
    def __init__(self, data_dir, image_dir="image", label_dir="label", transform=None, label_suffix='.tif',
                 normalize=v2.Normalize(
                     mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]
                 )):
        if not issubclass(type(transform), v2.Transform):
            raise TypeError('transform must be a subclass of torchvision.transforms.v2.Transform')
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_dir)
        self.label_dir = os.path.join(data_dir, label_dir)
        self.transform = transform
        self.label_suffix = label_suffix
        # 获取所有影像文件名（不含扩展名）
        self.ids = [f.split('.')[0] for f in os.listdir(self.image_dir)
                    if f.endswith('.tif')]
        self.normalize = normalize
        self.imagePreTransform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True)  # Scales values to [0.0, 1.0]
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # 加载影像（假设为 RGB，3 通道）
        img_path = os.path.join(self.image_dir, img_id + '.tif')
        # 使用 opencv 读取（统一以RGB格式读取）
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

        assert image.ndim == 3  # 确保数据是3通道彩色图，断言以防止错误数据被继续传递导致问题难以发觉
        # 加载标签
        label_path = os.path.join(self.label_dir, img_id + self.label_suffix)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label.ndim == 3 and label.shape[-1] == 1:
            label = np.squeeze(label)
        assert label.ndim == 2  # 确保标签是2通道(H,W)图
        # 将标签二值化为 0 和 1
        if label.dtype == np.uint8 and label.max() == 255:
            label = (label > 127).astype(np.uint8)
        else:
            label = (label > 0.5).astype(np.uint8)
        # 转为 PyTorch Tensor (C, H, W) ，opencv读取的是HWC
        image = TV.Image(self.imagePreTransform(image))
        label = TV.Mask(label)

        # 应用数据增强（同步处理图像和标签）
        if self.transform:
            image, label = self.transform(image, label)
            # 若没有增强，转为标准化流程
        if self.normalize:
            image = self.normalize(image)
        if len(label.shape) == 3:
            label = torch.squeeze(label, dim=0)

        return image, label.to(dtype=torch.int64)
