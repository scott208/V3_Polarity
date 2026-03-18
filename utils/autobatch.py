# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch
from torch.cuda import amp

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640):
    # 检查训练的批次大小
    with amp.autocast():  # 启用自动混合精度（自动进行半精度计算，以节省内存和加速训练）
        return autobatch(deepcopy(model).train(), imgsz)  # 计算最佳批次大小

def autobatch(model, imgsz=640, fraction=0.9, batch_size=16):
    # 自动估算最佳批次大小，以使用可用CUDA内存的`fraction`比例
    # 使用示例：
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov3', 'yolov3', autoshape=False)
    #     print(autobatch(model))

    prefix = colorstr('AutoBatch: ')  # 设置前缀，用于日志信息
    LOGGER.info(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')  # 打印计算批次大小的日志信息
    device = next(model.parameters()).device  # 获取模型所在的设备
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')  # 如果设备是CPU，使用默认批次大小
        return batch_size

    d = str(device).upper()  # 获取设备字符串表示（例如 'CUDA:0'）
    properties = torch.cuda.get_device_properties(device)  # 获取CUDA设备属性
    t = properties.total_memory / 1024 ** 3  # 设备总内存（以GiB为单位）
    r = torch.cuda.memory_reserved(device) / 1024 ** 3  # 设备上保留的内存（以GiB为单位）
    a = torch.cuda.memory_allocated(device) / 1024 ** 3  # 设备上已分配的内存（以GiB为单位）
    f = t - (r + a)  # 计算在保留内存中的可用内存
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')  # 打印设备内存信息

    batch_sizes = [1, 2, 4, 8, 16]  # 定义一组批次大小
    try:
        img = [torch.zeros(b, 3, imgsz, imgsz) for b in batch_sizes]  # 为每个批次大小创建一个零张量
        y = profile(img, model, n=3, device=device)  # 使用profile函数测量每种批次大小的内存使用情况
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')  # 捕获异常并打印警告信息

    y = [x[2] for x in y if x]  # 提取内存使用情况（第二个索引）
    batch_sizes = batch_sizes[:len(y)]  # 截取与内存使用情况相匹配的批次大小
    p = np.polyfit(batch_sizes, y, deg=1)  # 对批次大小和内存使用情况进行一次多项式拟合
    b = int((f * fraction - p[1]) / p[0])  # 根据拟合结果计算最佳批次大小
    LOGGER.info(f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%)')  # 打印最佳批次大小信息
    return b  # 返回计算出的最佳批次大小
