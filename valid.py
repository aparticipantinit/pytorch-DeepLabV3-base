import os
import sys
import cv2
from model import MyModelUtil
from config import args_c
from tabulate import tabulate
from tqdm import tqdm


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.dirname(__file__)
add_path(this_dir)

args = args_c('./my_model.yaml')
config = args.getConfig()
config.defrost()
config.TEST.BATCH_SIZE_PER_GPU = 32
config.freeze()
model = MyModelUtil(num_classes=2)
model.final_output_dir = r"output\crop_image\my_model"
model.set_config(config)
# raise SystemExit
model_path = r"output\crop_image\my_model\best_model.pth"
if os.path.exists(model_path):
    model.load_model(model_path)
    print("load model file:", model_path)
else:
    print("model file:", model_path, "不存在，正尝试加载之前的检查点")
    model.check_point(r"output\crop_image\my_model\checkpoint.pth")

# Data loading code
val_dir = os.path.join("./dataset", "test")
model.load_dataset(None, val_dir, config.MODEL.IMAGE_SIZE)

batch_time, losses, accuracy, iou, iou_0, iou_1 = model.validate_with_IOU()
row = [
    f'Test: Time {batch_time.avg:.3f}',
    f'Loss {losses.avg:.4f}',
    f'Error@acc {(100 - accuracy.avg):.4f} %',
    f'Accuracy {accuracy.avg:.3f} %',
    f'IOU {iou.avg:.4f} %',
    f'IOU_0 {(iou_0.avg * 100.0):.4f} %',
    f'IOU_1 {(iou_1.avg * 100.0):.4f} %',
]
headers_valid = [
    "AvgTime(s)",
    "AvgLoss", "Error@acc", "AvgAcc", "AvgIoU",
    "AvgIoU 0(background)", "AvgIoU 1(foreground)"
]
print(tabulate([row, ], headers=headers_valid, tablefmt="simple", showindex=False))

predict_dir = r".\dataset\test\image"
predict_save_dir = r".\predict"
if not os.path.exists(predict_save_dir):
    os.makedirs(predict_save_dir)
fs = os.listdir(predict_dir)

pbar = tqdm(fs,
            total=len(fs),
            desc=f"Predict test files:")
for i in pbar:
    p = os.path.join(predict_dir, i)
    mask = model.predict_file(p)
    p_o = os.path.join(predict_save_dir, i)
    cv2.imwrite(p_o, mask)

print(f"预测{len(fs)}张测试图像完成")
print("预测结果图像已保存至路径:", predict_dir)
# cv2.imshow("mask", mask)
# cv2.waitKey(0)
