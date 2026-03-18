# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import math
import random

import cv2
import numpy as np

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box
from utils.metrics import bbox_ioa


class Albumentations:
    # Albumentations类（可选，仅在包已安装时使用）
    def __init__(self):
        self.transform = None  # 初始化transform属性为None
        try:
            import albumentations as A  # 尝试导入albumentations库
            check_version(A.__version__, '1.0.3', hard=True)  # 检查albumentations库版本要求

            # 定义一系列数据增强变换
            self.transform = A.Compose([
                A.Blur(p=0.01),  # 模糊变换，概率为0.01
                A.MedianBlur(p=0.01),  # 中值模糊变换，概率为0.01
                A.ToGray(p=0.01),  # 转换为灰度图像，概率为0.01
                A.CLAHE(p=0.01),  # CLAHE（对比度限制自适应直方图均衡化），概率为0.01
                A.RandomBrightnessContrast(p=0.0),  # 随机亮度和对比度调整，概率为0.0
                A.RandomGamma(p=0.0),  # 随机Gamma调整，概率为0.0
                A.ImageCompression(quality_lower=75, p=0.0)  # 图像压缩，质量下限75，概率为0.0
            ],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))  # 传递YOLO格式的边界框参数

            # 打印已应用的变换信息
            LOGGER.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # 如果albumentations包未安装，忽略
            pass
        except Exception as e:
            # 打印任何其他异常
            LOGGER.info(colorstr('albumentations: ') + f'{e}')
    def __call__(self, im, labels, p=1.0):
        # 如果transform存在且随机数小于概率p，则应用变换
        if self.transform and random.random() < p:
            # 执行变换
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # 变换后的图像和边界框
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels  # 返回变换后的图像和标签


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV颜色空间增广
    if hgain or sgain or vgain:
        # 随机生成增益因子
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # 随机增益因子
        # 将图像从BGR转换为HSV颜色空间
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # 获取图像的数据类型，通常为uint8

        # 创建查找表（LUT）
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)  # 调整色调的查找表
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 调整饱和度的查找表
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 调整亮度的查找表

        # 应用查找表到HSV通道
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # 将图像从HSV转换回BGR，并直接更新输入图像
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # 不需要返回值


def hist_equalize(im, clahe=True, bgr=False):
    # 对BGR图像'im'进行直方图均衡化，图像形状为(n, m, 3)，像素范围为0-255
    # im: 输入图像，形状为(n, m, 3)
    # clahe: 是否使用CLAHE（对比度限制自适应直方图均衡化）
    # bgr: 输入图像是否为BGR格式（如果为False，则为RGB格式）

    # 将图像从BGR或RGB转换为YUV颜色空间
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)

    if clahe:
        # 创建CLAHE对象，clipLimit设置为2.0，tileGridSize设置为(8, 8)
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 应用CLAHE到Y通道（亮度通道）
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        # 如果不使用CLAHE，则直接对Y通道（亮度通道）进行直方图均衡化
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # 对Y通道直方图进行均衡化

    # 将YUV图像转换回BGR或RGB颜色空间
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # 将YUV图像转换为BGR或RGB

def replicate(im, labels):
    # 复制标签并在图像中创建标签副本
    h, w = im.shape[:2]  # 获取图像的高度和宽度
    boxes = labels[:, 1:].astype(int)  # 获取标签中的边界框坐标，并将其转换为整数
    x1, y1, x2, y2 = boxes.T  # 分别获取边界框的左上角和右下角坐标
    s = ((x2 - x1) + (y2 - y1)) / 2  # 计算边界框的边长（像素）

    # 对边界框按照边长排序，并选择最小的一半
    for i in s.argsort()[:round(s.size * 0.5)]:  # 选择最小的一半边界框
        x1b, y1b, x2b, y2b = boxes[i]  # 选择当前边界框的坐标
        bh, bw = y2b - y1b, x2b - x1b  # 计算边界框的高度和宽度
        # 随机生成偏移量，使新边界框在图像中不超出边界
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # 生成偏移量 x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]  # 计算新边界框的坐标
        # 将原边界框的内容复制到新的位置
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # 在新位置上复制图像内容
        # 将新的标签添加到标签列表中
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)
    return im, labels  # 返回修改后的图像和标签


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # 在满足步幅倍数约束的同时调整图像大小并填充
    shape = im.shape[:2]  # 当前图像的高度和宽度
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)  # 如果new_shape是整数，则将其转为元组形式

    # 计算缩放比例（新尺寸 / 旧尺寸）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 仅缩小，不放大（用于提高验证mAP）
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度和高度的缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 去除填充后的新尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算宽度和高度的填充量
    if auto:  # 自动填充为最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 使填充量为步幅的倍数
    elif scaleFill:  # 拉伸填充
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])  # 新尺寸为目标尺寸
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度和高度的比例

    dw /= 2  # 将填充量分成两边
    dh /= 2

    if shape[::-1] != new_unpad:  # 如果原图尺寸和新尺寸不同，则调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 填充图像边缘
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return im, ratio, (dw, dh)  # 返回调整后的图像、缩放比例和填充量


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # 随机透视变换函数（或仿射变换），用于数据增强
    # targets: 标签列表 [cls, xyxy]
    # segments: 轮廓分割（如果有的话）
    # degrees: 旋转角度范围
    # translate: 平移范围
    # scale: 缩放范围
    # shear: 剪切角度范围
    # perspective: 透视变换范围
    # border: 填充边界的宽度

    height = im.shape[0] + border[0] * 2  # 图像高度加上填充
    width = im.shape[1] + border[1] * 2  # 图像宽度加上填充

    # 中心平移矩阵
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x方向的平移（像素）
    C[1, 2] = -im.shape[0] / 2  # y方向的平移（像素）

    # 透视变换矩阵
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x方向透视（绕y轴）
    P[2, 1] = random.uniform(-perspective, perspective)  # y方向透视（绕x轴）

    # 旋转和缩放矩阵
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)  # 旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)  # 计算旋转矩阵

    # 剪切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x方向剪切（度）
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y方向剪切（度）

    # 平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x方向平移（像素）
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y方向平移（像素）

    # 合成变换矩阵（右到左的顺序非常重要）
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # 如果图像发生变化
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # 仿射变换
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # 可视化（可以解开注释以查看效果）
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # 基础图像
    # ax[1].imshow(im2[:, :, ::-1])  # 变换后的图像

    # 变换标签坐标
    n = len(targets)  # 标签数量
    if n:
        use_segments = any(x.any() for x in segments)  # 是否使用分割
        new = np.zeros((n, 4))  # 存储变换后的新边界框
        if use_segments:  # 变换分割区域
            segments = resample_segments(segments)  # 上采样
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # 变换
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # 透视缩放或仿射

                # 剪裁
                new[i] = segment2box(xy, width, height)

        else:  # 变换边界框
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # 变换
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # 透视缩放或仿射

            # 创建新的边界框
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # 剪裁边界框
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # 过滤候选框
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]
    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    # 实现 Copy-Paste 数据增强，参考 https://arxiv.org/abs/2012.07177
    # labels: 标签，nx5 np.array，包含 cls 和 xyxy
    # segments: 分割区域，包含每个标签的轮廓
    # p: 选择用于复制粘贴的比例

    n = len(segments)  # 轮廓的数量
    if p and n:
        h, w, c = im.shape  # 图像的高度、宽度和通道数
        im_new = np.zeros(im.shape, np.uint8)  # 创建一个全黑的图像，用于存放粘贴的区域

        # 随机选择要进行复制粘贴的区域
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]  # 获取标签和分割区域
            box = w - l[3], l[2], w - l[1], l[4]  # 计算粘贴区域的位置 (x1, y1, x2, y2)
            ioa = bbox_ioa(box, labels[:, 1:5])  # 计算与现有标签的交集面积
            if (ioa < 0.30).all():  # 允许现有标签被遮挡不超过 30%
                # 更新标签和分割区域
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))  # 更新分割区域（水平翻转）
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)  # 在黑色图像上绘制分割区域

        result = cv2.bitwise_and(src1=im, src2=im_new)  # 将原始图像与新图像进行按位与运算
        result = cv2.flip(result, 1)  # 水平翻转图像（增强分割区域）
        i = result > 0  # 获取要替换的像素
        # i[:, :] = result.max(2).reshape(h, w, 1)  # 在每个通道上操作（被注释掉的部分）
        im[i] = result[i]  # 替换图像中的像素

        # cv2.imwrite('debug.jpg', im)  # 用于调试，保存结果图像

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # 应用 Cutout 数据增强 https://arxiv.org/abs/1708.04552
    # 参数:
    #   im: 输入图像
    #   labels: 标签信息，包含类别和边界框的 np.array
    #   p: 执行 Cutout 的概率

    # 以概率 p 决定是否应用 Cutout 增强
    if random.random() < p:
        h, w = im.shape[:2]  # 图像的高度和宽度

        # 定义不同大小的遮挡区域的比例
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16

        for s in scales:
            # 随机生成遮挡区域的高度和宽度
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))

            # 随机确定遮挡区域的位置
            xmin = max(0, random.randint(0, w) - mask_w // 2)  # 左上角 x 坐标
            ymin = max(0, random.randint(0, h) - mask_h // 2)  # 左上角 y 坐标
            xmax = min(w, xmin + mask_w)  # 右下角 x 坐标
            ymax = min(h, ymin + mask_h)  # 右下角 y 坐标

            # 在遮挡区域内应用随机颜色的遮罩
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]  # 随机颜色值在 [64, 191] 范围内

            # 处理遮挡后的标签，移除被遮挡超过 60% 的标签
            if len(labels) and s > 0.03:  # 如果有标签且遮挡区域比例大于 0.03
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)  # 创建遮挡区域的边界框
                ioa = bbox_ioa(box, labels[:, 1:5])  # 计算遮挡区域与标签区域的交集面积
                labels = labels[ioa < 0.60]  # 移除遮挡面积大于 60% 的标签
    return labels


def mixup(im, labels, im2, labels2):
    # 应用 MixUp 数据增强 https://arxiv.org/pdf/1710.09412.pdf
    # 参数:
    #   im: 输入图像1
    #   labels: 输入图像1的标签信息，包含类别和边界框
    #   im2: 输入图像2
    #   labels2: 输入图像2的标签信息，包含类别和边界框

    # 计算 mixup 比率，beta 分布的参数为 32.0
    r = np.random.beta(32.0, 32.0)  # mixup 比率，alpha=beta=32.0

    # 使用 mixup 比率融合两张图像
    im = (im * r + im2 * (1 - r)).astype(np.uint8)

    # 合并两张图像的标签
    labels = np.concatenate((labels, labels2), 0)

    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    # 计算候选框: box1 为增强前的框，box2 为增强后的框
    # 参数:
    #   box1: 增强前的边界框，形状为 (4, n)，每列表示 [x1, y1, x2, y2]
    #   box2: 增强后的边界框，形状为 (4, n)，每列表示 [x1, y1, x2, y2]
    #   wh_thr: 宽高阈值（像素），用于过滤掉宽高小于该值的框
    #   ar_thr: 纵横比阈值，用于过滤掉纵横比超过该值的框
    #   area_thr: 面积比例阈值，用于过滤掉面积比小于该值的框
    #   eps: 避免除以零的极小值

    # 计算 box1 和 box2 的宽度和高度
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]  # box1 的宽度和高度
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]  # box2 的宽度和高度

    # 计算纵横比，防止除以零
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # 纵横比

    # 计算符合条件的候选框:
    # 1. box2 的宽度和高度大于 wh_thr
    # 2. box2 的面积比 (w2 * h2) 与 box1 的面积比 (w1 * h1) 大于 area_thr
    # 3. box2 的纵横比小于 ar_thr
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # 符合条件的候选框

