# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Activation functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# SiLU https://arxiv.org/pdf/1606.08415.pdf ----------------------------------------------------------------------------
class SiLU(nn.Module):  # 导出友好的 nn.SiLU() 版本
    @staticmethod
    def forward(x):
        # SiLU 激活函数的前向传播
        return x * torch.sigmoid(x)

class Hardswish(nn.Module):  # 导出友好的 nn.Hardswish() 版本
    @staticmethod
    def forward(x):
        # 使用 Hardtanh 近似 Hardswish 激活函数以兼容 TorchScript、CoreML 和 ONNX
        # return x * F.hardsigmoid(x)  # 对于 TorchScript 和 CoreML
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # 对于 TorchScript、CoreML 和 ONNX

# Mish https://github.com/digantamisra98/Mish --------------------------------------------------------------------------
class Mish(nn.Module):
    @staticmethod
    def forward(x):
        # Mish 激活函数: x 乘以 softplus(x) 的双曲正切
        return x * F.softplus(x).tanh()

class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # 保存输入 x 以供反向传播使用
            ctx.save_for_backward(x)
            # 计算 Mish 激活函数: x * tanh(ln(1 + exp(x)))
            return x.mul(torch.tanh(F.softplus(x)))

        @staticmethod
        def backward(ctx, grad_output):
            # 获取保存的输入 x
            x = ctx.saved_tensors[0]
            # 计算中间变量
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            # 计算梯度并返回
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        # 应用自定义的 Mish 激活函数
        return self.F.apply(x)



# FReLU https://arxiv.org/abs/2007.11824 -------------------------------------------------------------------------------
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # 输入通道数, 卷积核大小
        super().__init__()
        # 深度可分离卷积层: 每个输入通道有一个独立的卷积核
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        # 批量归一化层
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        # 计算深度可分离卷积后的结果，并应用批量归一化
        conv_output = self.bn(self.conv(x))
        # 取输入和卷积后的输出的最大值
        return torch.max(x, conv_output)

# ACON https://arxiv.org/pdf/2009.04759.pdf ----------------------------------------------------------------------------
class AconC(nn.Module):
    r""" ACON 激活函数（激活或不激活）。
    AconC: (p1*x - p2*x) * sigmoid(beta * (p1*x - p2*x)) + p2*x，其中 beta 是一个可学习的参数。
    参见论文 "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>。
    """

    def __init__(self, c1):
        super().__init__()
        # 初始化 p1 和 p2 为可学习的参数，形状为 (1, c1, 1, 1)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        # 初始化 beta 为可学习的参数，形状为 (1, c1, 1, 1)
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        # 计算 dpx = (p1 - p2) * x
        dpx = (self.p1 - self.p2) * x
        # 应用 ACON 激活函数
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x

class MetaAconC(nn.Module):
    r""" ACON 激活函数（激活或不激活）。
    MetaAconC: (p1*x - p2*x) * sigmoid(beta * (p1*x - p2*x)) + p2*x，其中 beta 由一个小网络生成。
    参见论文 "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>。
    """

    def __init__(self, c1, k=1, s=1, r=16):  # 输入通道数, 卷积核大小, 步幅, 维度压缩比例
        super().__init__()
        c2 = max(r, c1 // r)  # 计算中间通道数 c2
        # 初始化 p1 和 p2 为可学习的参数，形状为 (1, c1, 1, 1)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        # 定义两个卷积层，用于生成 beta 参数
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)
        # 自定义的 Batch Normalization 层已被注释掉
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        # 计算输入特征图 x 的均值，作为生成 beta 的输入
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # 计算 beta 参数，去掉了 Batch Normalization 层以修复稳定性问题
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # 使用小网络生成 beta 参数
        # 计算 ACON 激活函数
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x