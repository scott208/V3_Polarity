# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math

import numpy as np
import torch
import torch.nn as nn

from models.common import Conv
from utils.downloads import attempt_download


class CrossConv(nn.Module):
    # 交叉卷积下采样
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # 输入通道数, 输出通道数, 卷积核大小, 步幅, 组卷积, 扩展因子, 是否使用快捷连接
        super().__init__()
        c_ = int(c2 * e)  # 计算隐藏通道数
        self.cv1 = Conv(c1, c_, (1, k), (1, s))  # 第一层卷积，1xk 卷积核，步幅为 1x s
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)  # 第二层卷积，kx1 卷积核，步幅为 s x 1，支持组卷积
        self.add = shortcut and c1 == c2  # 如果 shortcut 为 True 且输入输出通道数相同，则启用快捷连接

    def forward(self, x):
        # 如果启用快捷连接，则将输入 x 和卷积输出相加；否则，仅返回卷积输出
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class Sum(nn.Module):
    # 对 2 个或更多层进行加权和 https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: 输入的层数
        super().__init__()
        self.weight = weight  # 是否应用权重的布尔值
        self.iter = range(n - 1)  # 迭代对象
        if weight:
            # 如果应用权重，创建权重参数
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # 层权重，初始为负数，并允许梯度计算

    def forward(self, x):
        y = x[0]  # 初始输出为输入的第一个层
        if self.weight:
            # 如果使用权重，对权重进行 sigmoid 激活和缩放
            w = torch.sigmoid(self.w) * 2
            # 按权重对后续层进行加权和
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            # 如果不使用权重，直接将所有输入层进行求和
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # 混合深度卷积 https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # 输入通道数, 输出通道数, 卷积核大小, 步幅, 通道策略
        super().__init__()
        n = len(k)  # 卷积的数量

        if equal_ch:  # 如果 equal_ch 为 True，确保每个组的通道数相等
            i = torch.linspace(0, n - 1E-6, c2).floor()  # 计算输出通道的索引
            c_ = [(i == g).sum() for g in range(n)]  # 计算每个卷积核对应的通道数
        else:  # 如果 equal_ch 为 False，确保每个组的权重数量相等
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)  # 构造矩阵 a
            a -= np.roll(a, 1, axis=1)  # 计算差分
            a *= np.array(k) ** 2  # 根据卷积核大小调整
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # 解方程组，获得每个卷积核的通道数

        # 定义多个卷积层，每个卷积层的通道数由 c_ 决定
        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)  # 批归一化层
        self.act = nn.SiLU()  # 激活函数

    def forward(self, x):
        # 对每个卷积层进行前向传播，将结果拼接在一起，并通过批归一化和激活函数
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # 模型集成
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            # 对每个模型进行前向传播，获取输出并追加到 y 列表
            y.append(module(x, augment, profile, visualize)[0])

        # 下面的代码可以选择集成策略：
        # y = torch.stack(y).max(0)[0]  # 最大值集成
        # y = torch.stack(y).mean(0)  # 平均值集成
        y = torch.cat(y, 1)  # 拼接集成
        return y, None  # 返回集成后的结果和 None（用于推理，训练输出）


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from models.yolo import Detect, Model

    # 加载模型权重，支持单个模型或多个模型的权重列表
    # weights 可以是单个权重文件的路径，也可以是包含多个权重文件路径的列表
    model = Ensemble()  # 创建一个模型集成对象
    for w in weights if isinstance(weights, list) else [weights]:
        # 加载权重文件
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # 下载并加载权重
        if fuse:
            # 将模型的卷积层和批量归一化层融合，并设置为评估模式
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 模型
        else:
            # 仅加载模型权重，不进行融合
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # 不融合

    # 兼容性更新
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # 设置是否使用原地操作以兼容 PyTorch 1.7.0
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # 新版 Detect 层的兼容性
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)  # 重新设置 anchor_grid
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # 兼容 PyTorch 1.6.0

    if len(model) == 1:
        return model[-1]  # 如果只有一个模型，返回该模型
    else:
        print(f'Ensemble created with {weights}\n')  # 打印模型集成信息
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))  # 从最后一个模型中复制 'names' 属性
        # 设置集成模型的步幅为所有模型中的最大步幅
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride
        return model  # 返回模型集成

