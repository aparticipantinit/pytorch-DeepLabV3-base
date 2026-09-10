import torch


def pixel_accuracy(output, target):
    """
    计算像素准确率（Pixel Accuracy）
    output: (N, C, H, W)  模型输出的 logits ,批次为N，实际大小是(N, 2, H, W)
    target: (N, H, W)     标签索引
    返回: 准确率（百分比，标量）
    """
    with torch.no_grad():
        # 获取预测类别 (N, H, W)
        pred = torch.argmax(output, dim=1)
        # 计算正确像素数
        correct = (pred == target).sum().item()
        total = target.numel()
        acc = 100.0 * correct / total
    return acc


def mean_iou(output, target, num_classes=2):
    """
    计算平均交并比（Mean IoU），返回百分比数值。

    Args:
        output (torch.Tensor): 模型输出的 logits，形状为 (N, C, H, W)
        target (torch.Tensor): 标签索引，形状为 (N, H, W)，取值为 0 ~ C-1
        num_classes (int): 类别总数，默认 2（二分类）

    Returns:
        float: 平均 IoU（百分比），标量。若所有类别均未出现，则返回 0.0。
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)  # (N, H, W)

        iou_list = []
        for cls in range(num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)

            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()

            if union == 0:
                # 如果该类在预测和真值中均未出现，则忽略（不计入平均）
                continue

            iou = intersection / union
            iou_list.append(iou.item())

        if not iou_list:
            return 0.0

        _mean_iou = sum(iou_list) / len(iou_list)
        return 100.0 * _mean_iou


def accuracy_and_iou(output, target, num_classes=2):
    _mean_iou = 0
    iou_list = []
    with torch.no_grad():
        # 获取预测类别 (N, H, W)
        pred = torch.argmax(output, dim=1)
        # 计算正确像素数
        correct = (pred == target).sum().item()
        total = target.numel()
        acc = 100.0 * correct / total

        for cls in range(num_classes):
            pred_mask = (pred == cls)
            target_mask = (target == cls)

            TP = (pred_mask & target_mask).sum().item()
            FP = (pred_mask & ~target_mask).sum().item()  # 预测为c，但真实不是c
            FN = (~pred_mask & target_mask).sum().item()  # 真实是c，但预测不是c

            union = TP + FP + FN
            if union == 0:
                iou = 1.0
            else:
                iou = TP / union
            iou_list.append(iou)

        if not iou_list:
            _mean_iou = 0.0
        else:
            _mean_iou = 100.0 * sum(iou_list) / len(iou_list)
    return acc, _mean_iou, iou_list
