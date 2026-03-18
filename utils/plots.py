# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Plotting utils
"""

import math
import os
from copy import copy
from pathlib import Path

import cv2  # OpenCV 库，用于图像处理
import matplotlib
import matplotlib.pyplot as plt  # Matplotlib 库，用于绘图
import numpy as np  # NumPy 库，用于数值计算
import pandas as pd  # Pandas 库，用于数据操作
import seaborn as sn  # Seaborn 库，用于数据可视化
import torch  # PyTorch 库，用于机器学习和张量计算
from PIL import Image, ImageDraw, ImageFont  # Pillow 库，用于图像处理

from utils.general import (LOGGER, Timeout, check_requirements, clip_coords, increment_path, is_ascii, is_chinese,
                           try_except, user_config_dir, xywh2xyxy, xyxy2xywh)  # 导入 utils.general 模块中的函数和类
from utils.metrics import fitness  # 导入 utils.metrics 模块中的 fitness 函数

# 配置
CONFIG_DIR = user_config_dir()  # 获取 Ultralytics 设置目录路径
RANK = int(os.getenv('RANK', -1))  # 从环境变量中获取 RANK 值，默认为 -1
matplotlib.rc('font', **{'size': 11})  # 设置 Matplotlib 的默认字体大小
matplotlib.use('Agg')  # 使用 'Agg' 后端以便将绘图保存为文件，而不是在屏幕上显示


class Colors:
    """
    代码定义了一个 Colors 类，用于管理和使用一组预定义的颜色。这些颜色可以用来为图像处理或数据可视化任务着色。
    """
    # Ultralytics 调色板 https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        # 将 hex 颜色码转换为 RGB 格式并存储在调色板中
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)  # 调色板中颜色的数量

    def __call__(self, i, bgr=False):
        # 获取调色板中第 i 个颜色，支持 BGR 顺序（用于 OpenCV）
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # 将 hex 颜色码转换为 RGB 顺序（PIL 格式）
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # 创建 Colors 类的实例，用于 'from utils.plots import colors'



def check_font(font='Arial.ttf', size=10):
    # 返回一个 PIL TrueType 字体对象，如果必要会从 CONFIG_DIR 下载
    font = Path(font)
    font = font if font.exists() else (CONFIG_DIR / font.name)  # 检查字体文件是否存在，否则从 CONFIG_DIR 路径下查找同名文件
    try:
        return ImageFont.truetype(str(font) if font.exists() else font.name, size)  # 尝试加载字体
    except Exception as e:  # 如果字体文件不存在，则下载字体文件
        url = "https://ultralytics.com/assets/" + font.name
        print(f'Downloading {url} to {font}...')
        torch.hub.download_url_to_file(url, str(font), progress=False)  # 从 URL 下载字体文件到指定路径
        try:
            return ImageFont.truetype(str(font), size)  # 再次尝试加载下载的字体文件
        except TypeError:
            check_requirements('Pillow>=8.4.0')  # 如果加载失败，检查 Pillow 版本要求


class Annotator:
    if RANK in (-1, 0):
        check_font()  # 如果 RANK 是 -1 或 0，则下载字体文件（如果需要）

    # Annotator 用于训练/验证集的马赛克和 jpg 图像以及检测/中心点预测推断注释
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        if self.pil:  # 使用 PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                   size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # 使用 cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # 线条宽度

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # 给图像添加一个带标签的 xyxy 矩形框
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # 绘制矩形框
            if label:
                w, h = self.font.getsize(label)  # 获取文本宽度和高度
                outside = box[1] - h >= 0  # 判断标签是否超出矩形框外
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # 适用于 PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # 使用 cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # 字体粗细
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # 获取文本宽度和高度
                outside = p1[1] - h - 3 >= 0  # 判断标签是否超出矩形框外
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # 填充矩形框
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # 给图像添加一个矩形框（仅适用于 PIL）
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255)):
        # 给图像添加文本（仅适用于 PIL）
        w, h = self.font.getsize(text)  # 获取文本宽度和高度
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    def result(self):
        # 将带有注释的图像作为数组返回
        return np.asarray(self.im)



def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
        x: 要可视化的特征图张量。
        module_type: 模块类型，用于区分不同类型的层。
        stage: 模型中的阶段或层索引。
        n: 要绘制的特征图的最大数量。
        save_dir: 保存图像的目录路径。
    """
    # 检查是否为 Detect 层：
    # 如果当前层的类型不包含 'Detect'，则进行可视化操作。
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # 获取特征图张量的维度：批次大小、通道数、高度和宽度。
        if height > 1 and width > 1:  # 只有在高度和宽度大于1时才进行处理。
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # 生成文件名,根据当前层的阶段和类型生成文件名。
            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # 选择批次中的第一个样本，将特征图按通道数分块。
            n = min(n, channels)  # 确定要绘制的特征图数量。

            # 创建一个包含特征图子图的图形对象，按 8 列和适当的行数排列子图。
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            # 将每个特征图块绘制到子图中，并关闭坐标轴显示。
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            # 打印保存信息，将图形保存到指定目录，并关闭图形对象。
            print(f'Saving {save_dir / f}... ({n}/{channels})')
            plt.savefig(save_dir / f, dpi=300, bbox_inches='tight')
            plt.close()


def hist2d(x, y, n=100):
    """
    函数 hist2d 用于生成一个二维直方图
    """
    # 用于生成 labels.png 和 evolve.png 的二维直方图
    # 生成等间距的边界，用于划分 x 和 y 轴上的区间
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    # 计算二维直方图，hist 是频数矩阵，xedges 和 yedges 是每个区间的边界
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    # 找到每个 x 值对应的 bin 索引，并确保索引在有效范围内
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    # 找到每个 y 值对应的 bin 索引，并确保索引在有效范围内
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    # 返回每个 (x, y) 对应 bin 的对数频数
    return np.log(hist[xidx, yidx])

def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    """
    用于对数据应用巴特沃斯低通滤波器，并通过前向-后向滤波（filtfilt）消除相位延迟。
    """
    # 导入必要的函数
    from scipy.signal import butter, filtfilt
    # 低通滤波器设计函数
    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs  # 计算奈奎斯特频率（采样频率的一半）
        normal_cutoff = cutoff / nyq  # 归一化截止频率
        return butter(order, normal_cutoff, btype='low', analog=False)  # 设计巴特沃斯低通滤波器
    # 获取巴特沃斯低通滤波器的系数
    b, a = butter_lowpass(cutoff, fs, order=order)
    # 使用前向-后向滤波器进行滤波，避免相位延迟
    return filtfilt(b, a, data)


def output_to_target(output):
    """
    用于将模型输出的检测结果转换为目标格式，包含批次 ID、类别 ID、中心坐标、宽度、高度和置信度。
    """
    # 将模型输出转换为目标格式 [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():  # 提取每个检测框的坐标、置信度和类别
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])  # 转换坐标格式并添加到目标列表
    return np.array(targets)  # 返回目标数组



def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=1920, max_subplots=16):
    """
    这个函数主要用于目标检测任务中的图像可视化，特别是用于展示检测结果。
    通过将多张图像和对应的检测结果绘制在一个网格中，便于快速浏览和评估模型的检测效果。
    """

    # 绘制带有标签的图像网格

    # 检查输入数据类型并转换为 numpy 数组
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if np.max(images[0]) <= 1:
        images *= 255  # 反归一化（如果图像像素值在 [0, 1] 之间）

    # 获取批次大小、图像高度和宽度
    bs, _, h, w = images.shape  # 批次大小, _, 高度, 宽度
    bs = min(bs, max_subplots)  # 限制绘图图像数量
    ns = np.ceil(bs ** 0.5)  # 子图数量（取平方根后向上取整）

    # 构建初始空白图像（马赛克图像）
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # 初始化全白图像

    # 将每张图像放置到马赛克图像中
    for i, im in enumerate(images):
        if i == max_subplots:  # 如果最后一个批次的图像数量少于预期
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 块的原点坐标
        im = im.transpose(1, 2, 0)  # 转置图像以匹配 (height, width, channels) 格式
        mosaic[y:y + h, x:x + w, :] = im  # 将图像复制到马赛克图像中

    # 可选：调整图像大小
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # 标注图像
    fs = int((h + w) * ns * 0.01)  # 字体大小
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True)
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # 块的原点坐标
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # 画出边框
        if paths:
            annotator.text((x + 5, y + 5 + h), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # 文件名
        if len(targets) > 0:
            ti = targets[targets[:, 0] == i]  # 选择当前图像的目标
            boxes = xywh2xyxy(ti[:, 2:6]).T  # 将目标坐标从 xywh 转换为 xyxy
            classes = ti[:, 1].astype('int')  # 获取目标类别
            labels = ti.shape[1] == 6  # 检查是否有置信度列
            conf = None if labels else ti[:, 6]  # 获取置信度（如果有）

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # 如果坐标已归一化
                    boxes[[0, 2]] *= w  # 按比例缩放到像素
                    boxes[[1, 3]] *= h
                elif scale < 1:  # 如果是绝对坐标且图像缩放，则按比例缩放
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)  # 获取颜色
                cls = names[cls] if names else cls  # 获取类别名称
                if labels or conf[j] > 0.25:  # 0.25 置信度阈值
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'  # 标签文本
                    annotator.box_label(box, label, color=color)  # 添加标签
    annotator.im.save(fname)  # 保存图像


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    """
    用于绘制学习率（LR）的变化曲线，模拟训练过程中的学习率变化。
    """
    # 绘制学习率（LR），模拟完整的训练过程
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # 复制优化器和调度器，以不修改原始对象
    y = []  # 用于存储每个 epoch 的学习率
    for _ in range(epochs):
        scheduler.step()  # 更新调度器，计算新的学习率
        y.append(optimizer.param_groups[0]['lr'])  # 获取当前学习率并存储

    # 绘制学习率变化曲线
    plt.plot(y, '.-', label='LR')  # 绘制学习率曲线
    plt.xlabel('epoch')  # x 轴标签
    plt.ylabel('LR')  # y 轴标签
    plt.grid()  # 显示网格
    plt.xlim(0, epochs)  # 设置 x 轴范围
    plt.ylim(0)  # 设置 y 轴范围

    # 保存图像
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)  # 保存为 LR.png
    plt.close()  # 关闭当前图像



def plot_val_txt():  # 从 utils.plots 导入 *; plot_val()
    """
    这个函数 plot_val_txt 用于从 val.txt 文件中读取数据并绘制直方图
    """
    # 绘制 val.txt 的直方图
    x = np.loadtxt('val.txt', dtype=np.float32)  # 从 val.txt 文件加载数据
    box = xyxy2xywh(x[:, :4])  # 将边界框坐标从 xyxy 转换为 xywh 格式
    cx, cy = box[:, 0], box[:, 1]  # 提取中心坐标 (cx, cy)

    # 绘制二维直方图
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)  # 绘制中心坐标的二维直方图
    ax.set_aspect('equal')  # 设置 x 和 y 轴比例相等
    plt.savefig('hist2d.png', dpi=300)  # 保存二维直方图为 hist2d.png

    # 绘制一维直方图
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)  # 绘制 cx 的一维直方图
    ax[1].hist(cy, bins=600)  # 绘制 cy 的一维直方图
    plt.savefig('hist1d.png', dpi=200)  # 保存一维直方图为 hist1d.png



def plot_targets_txt():  # 从 utils.plots 导入 *; plot_targets_txt()
    """
    这个函数 plot_targets_txt 用于从 targets.txt 文件中读取数据并绘制目标的直方图。
    """
    # 绘制 targets.txt 的直方图
    x = np.loadtxt('targets.txt', dtype=np.float32).T  # 从 targets.txt 文件加载数据并转置
    s = ['x targets', 'y targets', 'width targets', 'height targets']  # 目标的名称列表

    # 创建 2x2 的子图
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()  # 将二维数组展平成一维数组，以便于遍历

    # 绘制每个目标的直方图
    for i in range(4):
        ax[i].hist(x[i], bins=100, label=f'{x[i].mean():.3g} +/- {x[i].std():.3g}')  # 绘制直方图并显示均值和标准差
        ax[i].legend()  # 显示图例
        ax[i].set_title(s[i])  # 设置每个子图的标题

    plt.savefig('targets.jpg', dpi=200)  # 保存直方图为 targets.jpg


def plot_val_study(file='', dir='', x=None):  # 从 utils.plots 导入 *; plot_val_study()
    """
    这个函数 plot_val_study 用于从 study.txt 文件绘制验证结果，
    或者从指定目录中绘制所有 study*.txt 文件的结果。
    """
    # 绘制由 val.py 生成的 study.txt 文件（或绘制目录下所有 study*.txt 文件）
    save_dir = Path(file).parent if file else Path(dir)  # 确定保存目录
    plot2 = False  # 是否绘制额外的结果
    if plot2:
        ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()  # 创建额外的子图
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)  # 创建主图
    # 遍历所有 study*.txt 文件
    for f in sorted(save_dir.glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T  # 加载数据并转置
        x = np.arange(y.shape[1]) if x is None else np.array(x)  # 确定 x 轴数据
        if plot2:
            s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_preprocess (ms/img)', 't_inference (ms/img)', 't_NMS (ms/img)']
            for i in range(7):
                ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)  # 绘制额外的结果
                ax[i].set_title(s[i])  # 设置标题
        j = y[3].argmax() + 1  # 找到最大 mAP@.5 的索引
        ax2.plot(y[5, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))  # 绘制主图的 mAP 数据

    # 绘制 EfficientDet 的参考线
    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    # 设置图表的样式和标签
    ax2.grid(alpha=0.2)  # 添加网格
    ax2.set_yticks(np.arange(20, 60, 5))  # 设置 y 轴刻度
    ax2.set_xlim(0, 57)  # 设置 x 轴范围
    ax2.set_ylim(25, 55)  # 设置 y 轴范围
    ax2.set_xlabel('GPU Speed (ms/img)')  # x 轴标签
    ax2.set_ylabel('COCO AP val')  # y 轴标签
    ax2.legend(loc='lower right')  # 显示图例

    f = save_dir / 'study.png'  # 保存图像的路径
    print(f'Saving {f}...')
    plt.savefig(f, dpi=300)  # 保存图像

@try_except  # known issue https://github.com/ultralytics/yolov5/issues/5395
@Timeout(30)  # known issue https://github.com/ultralytics/yolov5/issues/5611
def plot_labels(labels, names=(), save_dir=Path('')):
    # 绘制数据集标签
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")

    # 提取类别和框信息
    c, b = labels[:, 0], labels[:, 1:].transpose()  # c: 类别, b: 框
    nc = int(c.max() + 1)  # 类别数量
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # Seaborn相关图
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # Matplotlib标签分布图
    matplotlib.use('svg')  # 使用svg格式以加快绘制速度
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()

    # 绘制类别实例直方图
    y = ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')  # y轴标签

    # 设置x轴标签
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')

    # 绘制框中心和尺寸的直方图
    sn.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sn.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # 绘制矩形框
    labels[:, 1:3] = 0.5  # 中心位置
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000  # 转换为x1, y1, x2, y2格式
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)  # 创建白色背景图像

    # 绘制前1000个框
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # 绘制框

    ax[1].imshow(img)  # 显示图像
    ax[1].axis('off')  # 关闭坐标轴

    # 隐藏图形边框
    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)  # 保存标签图
    matplotlib.use('Agg')  # 切换回Agg模式
    plt.close()  # 关闭图形


def plot_evolve(evolve_csv='path/to/evolve.csv'):  # from utils.plots import *; plot_evolve()
    # 绘制 evolve.csv 的超参数进化结果
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # 最大适应度的索引

    # 设置绘图区域大小
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})

    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # 最佳单次结果
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})  # 标题限制为 40 个字符
        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')

    f = evolve_csv.with_suffix('.png')  # 文件名
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')



def plot_results(file='path/to/results.csv', dir=''):
    """
    这个函数 plot_results 用于从指定的 CSV 文件绘制训练过程中的结果。
    """
    # 绘制训练结果的 results.csv 文件。用法: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)  # 确定保存目录
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)  # 创建 2x5 的子图
    ax = ax.ravel()  # 将二维数组展平成一维数组
    files = list(save_dir.glob('results*.csv'))  # 查找目录中所有以 results 开头的 CSV 文件
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'  # 确保找到文件

    # 遍历所有找到的文件
    for fi, f in enumerate(files):
        try:
            data = pd.read_csv(f)  # 读取 CSV 文件
            s = [x.strip() for x in data.columns]  # 获取列名并去除空格
            x = data.values[:, 0]  # x 轴数据（通常为训练轮次或步骤）
            # 绘制每个指标的图表
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j]  # 获取 y 轴数据
                # y[y == 0] = np.nan  # 不显示零值（可选）
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)  # 绘制曲线
                ax[i].set_title(s[j], fontsize=12)  # 设置标题
                # if j in [8, 9, 10]:  # 共享训练和验证损失的 y 轴（可选）
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')  # 捕获并打印错误信息
    ax[1].legend()  # 显示图例
    fig.savefig(save_dir / 'results.png', dpi=200)  # 保存结果图像
    plt.close()  # 关闭绘图


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    """
    这个函数 profile_idetection 用于绘制 iDetection 的每图像日志
    """
    # 绘制 iDetection 的每图像日志 '*.txt' 文件。用法: from utils.plots import *; profile_idetection()
    ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)[1].ravel()  # 创建 2x4 的子图
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)',
         'real-world FPS']  # 子图标题
    files = list(Path(save_dir).glob('frames*.txt'))  # 查找所有以 frames 开头的 txt 文件

    # 遍历所有找到的文件
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # 加载数据并去除前后不需要的行
            n = results.shape[1]  # 获取数据行数
            x = np.arange(start, min(stop, n) if stop else n)  # 确定绘图的 x 轴范围
            results = results[:, x]  # 根据 x 轴范围选择数据
            t = (results[0] - results[0].min())  # 设置 t0=0s
            results[0] = x  # 将 x 轴数据设置为索引

            # 绘制每个指标的图表
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')  # 获取图例标签
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)  # 绘制曲线
                    a.set_title(s[i])  # 设置子图标题
                    a.set_xlabel('time (s)')  # 设置 x 轴标签

                    # 隐藏顶部和右侧边框
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()  # 如果没有足够的数据，则移除子图
        except Exception as e:
            print(f'Warning: Plotting error for {f}; {e}')  # 捕获并打印错误信息

    ax[1].legend()  # 显示图例
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)  # 保存结果图像


def save_one_box(xyxy, im, file='image.jpg', gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    这个函数 save_one_box 的主要功能是从给定的图像中裁剪出指定边界框的部分
    """
    # 将图像裁剪并保存为指定文件，裁剪尺寸为原框的 {gain} 倍加上 {pad} 像素
    xyxy = torch.tensor(xyxy).view(-1, 4)  # 将输入框转换为张量，并重塑形状为 (N, 4)
    b = xyxy2xywh(xyxy)  # 将 xyxy 格式转换为 xywh 格式的边界框

    if square:
        # 如果需要将矩形框调整为正方形，取宽和高的最大值
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # 尝试将矩形调整为正方形

    # 计算新的边界框尺寸
    b[:, 2:] = b[:, 2:] * gain + pad  # 宽高乘以增益并加上填充
    xyxy = xywh2xyxy(b).long()  # 转换回 xyxy 格式并转换为整数

    clip_coords(xyxy, im.shape)  # 限制坐标在图像尺寸范围内
    # 裁剪图像
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]

    if save:
        # 如果需要保存裁剪的图像
        file.parent.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
        cv2.imwrite(str(increment_path(file).with_suffix('.jpg')), crop)  # 保存裁剪图像
    return crop  # 返回裁剪后的图像

