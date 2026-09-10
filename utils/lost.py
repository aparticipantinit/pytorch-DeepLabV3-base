import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        pred_softmax = F.softmax(predictions, dim=1)  # [B, 2, H, W]
        pred_building = pred_softmax[:, 1, :, :]  # [B, H, W] 提取建筑通道

        B = pred_building.size(0)
        pred_flat = pred_building.view(B, -1)
        target_flat = targets.view(B, -1).float()  # 转为0/1浮点数

        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5, dice_smooth=1e-7, cross_entropy_weight=torch.tensor([1.0, 3.0])):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.ce = nn.CrossEntropyLoss(weight=cross_entropy_weight, reduction="mean")
        self.dice_smooth = dice_smooth
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(self, predictions, targets):
        loss_ce = self.ce(predictions, targets)
        loss_dice = self.dice(predictions, targets)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
