# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch utils
"""
import datetime
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils.general import LOGGER

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    这个函数 torch_distributed_zero_first 的主要功能是作为一个装饰器，用于分布式训练中的进程同步。
    装饰器，使分布式训练中的所有进程在本地主进程完成某项操作之前等待。
    参数:
    - local_rank (int): 当前进程的本地排名（0表示主进程，-1表示单进程模式）。
    """
    if local_rank not in [-1, 0]:
        # 如果当前进程不是主进程，等待主进程完成操作
        dist.barrier(device_ids=[local_rank])
    yield  # 暂停执行，允许主进程完成操作
    if local_rank == 0:
        # 如果当前进程是主进程，等待所有进程同步
        dist.barrier(device_ids=[0])

def date_modified(path=__file__):
    """
    这个函数 date_modified 的主要功能是获取指定文件的最后修改日期，并以人类可读的格式返回。
    """
    # 返回可读的文件修改日期，格式为 'YYYY-MM-DD'
    # 获取文件的最后修改时间戳
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    # 格式化日期并返回
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):  # path 必须是一个目录
    """
    这个函数 git_describe 的主要功能是获取指定目录下 Git 仓库的描述信息。
    """
    # 返回可读的 git 描述信息，例如 'v5.0-5-g3e25f1e'
    # 参考文档: https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'  # 构建 git 命令
    try:
        # 执行命令并获取输出
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # 如果不是一个 git 仓库，返回空字符串


# 函数接收三个参数：device、batch_size和newline。
def select_device(device='', batch_size=None, newline=True):
    # device = 'cpu' 或 '0' 或 '0,1,2,3'
    s = f'YOLOv3 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # 它初始化一个包含环境初始信息的字符串s，包括YOLOv3版本和PyTorch版本。
    device = str(device).strip().lower().replace('cuda:', '')  # 将 device 参数转换为小写字符串，并移除 cuda: 前缀。

    # 判断是否是 CPU 设备
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  #如果是，则设置环境变量 CUDA_VISIBLE_DEVICES 为 -1，强制禁用CUDA。
    elif device:  # 如果请求非 CPU 设备，则设置环境变量 CUDA_VISIBLE_DEVICES 为指定设备，并检查 CUDA 是否可用。
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    #  如果使用CUDA设备，获取设备列表。
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # 范围(torch.cuda.device_count())  # 如 0,1,6,7
        n = len(devices)  # 设备数量
        if n > 1 and batch_size:   # 检查 batch_size 是否是设备数量的倍数
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        # 遍历设备列表，获取每个设备的属性，并将其信息添加到字符串 s 中。
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    # 如果 newline为False，去掉字符串s末尾的换行符。
    if not newline:
        s = s.rstrip()
    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def time_sync():
    """
    这个函数 time_sync 的主要功能是获取一个精确的当前时间。
    """
    # 返回精确的当前时间（以秒为单位）
    # 如果可用，首先同步 CUDA 设备，以确保时间测量的准确性

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待所有CUDA操作完成
    return time.time()  # 返回当前时间


def profile(input, ops, n=10, device=None):
    """
    该函数用于分析给定输入和模型操作的性能，记录模型的参数数量、GFLOPs（每秒十亿次浮点运算）、GPU内存占用、前向和反向传播的平均时间。
    """
    # 速度/内存/FLOPs 分析器
    #
    # 用法示例：
    #     input = torch.randn(16, 3, 640, 640)  # 生成随机输入
    #     m1 = lambda x: x * torch.sigmoid(x)  # 示例操作1
    #     m2 = nn.SiLU()  # 示例操作2
    #     profile(input, [m1, m2], n=100)  # 在100次迭代中进行分析

    results = []  # 存储分析结果
    device = device or select_device()  # 选择使用的设备（GPU/CPU）

    # 打印表头
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>24s}")

    # 确保输入为列表
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)  # 将输入移动到设备
        x.requires_grad = True  # 需要计算梯度

        # 确保操作为列表
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # 将模型移动到设备
            # 如果使用半精度且输入为float16，转换模型为半精度
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m

            tf, tb, t = 0, 0, [0, 0, 0]  # 初始化前向和反向传播时间

            try:
                # 计算GFLOPs
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2
            except:
                flops = 0  # 如果失败，GFLOPs设为0

            try:
                # 多次运行以获取平均时间
                for _ in range(n):
                    t[0] = time_sync()  # 记录前向传播开始时间
                    y = m(x)  # 前向传播
                    t[1] = time_sync()  # 记录前向传播结束时间
                    try:
                        # 计算反向传播
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()  # 记录反向传播结束时间
                    except Exception:  # 如果没有反向传播方法
                        t[2] = float('nan')  # 记录为NaN

                    # 计算每次前向传播和反向传播的平均时间
                    tf += (t[1] - t[0]) * 1000 / n  # 前向传播时间（毫秒）
                    tb += (t[2] - t[1]) * 1000 / n  # 反向传播时间（毫秒）

                # 获取GPU内存使用情况（GB）
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
                s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'  # 输入形状
                s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'  # 输出形状
                # 计算模型参数总数
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0

                # 打印分析结果
                print(f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out])  # 保存结果
            except Exception as e:
                print(e)  # 打印错误信息
                results.append(None)  # 记录结果为None
            torch.cuda.empty_cache()  # 清理缓存以释放内存

    return results  # 返回结果列表


def is_parallel(model):
    # 如果模型是 DataParallel（DP）或 DistributedDataParallel（DDP）类型，则返回 True
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # 将模型去并行化：如果模型是 DataParallel（DP）或 DistributedDataParallel（DDP）类型，则返回单GPU模型
    return model.module if is_parallel(model) else model

# 用于初始化模型中的权重和偏置。该函数会遍历模型的所有模块.
# 根据模块的类型应用不同的初始化策略。
def initialize_weights(model):
    # model.modules()返回模型中所有模块的迭代器。
    # m代表当前遍历到的模块。
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            # 注释中建议使用 Kaiming正态分布初始化权重。
            # 这里实际的初始化方法被注释掉了，可以根据需要取消注释以应用该初始化方法。
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            # 对于 nn.BatchNorm2d层，设置eps和momentum参数。
            # eps 是一个小数值，防止在计算过程中出现除以零的情况，默认值通常是1e-5。
            #  momentum是用于运行时均值和方差计算的动量，默认值通常是0.1。
            m.eps = 1e-3
            m.momentum = 0.03
        # 对于这些激活函数层，设置inplace 参数为True。
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def find_modules(model, mclass=nn.Conv2d):
    # 找到与模块类 'mclass' 匹配的层索引
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]

def sparsity(model):
    # 返回模型的全局稀疏性
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()  # 累计参数总数
        b += (p == 0).sum()  # 累计为零的参数数量
    return b / a  # 返回稀疏性比例

def prune(model, amount=0.3):
    # 对模型进行剪枝，以达到请求的全局稀疏性
    import torch.nn.utils.prune as prune
    print('正在剪枝模型... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):  # 只对卷积层进行剪枝
            prune.l1_unstructured(m, name='weight', amount=amount)  # 进行L1无结构剪枝
            prune.remove(m, 'weight')  # 使剪枝结果永久生效
    print(' %.3g 全局稀疏性' % sparsity(model))  # 打印剪枝后的全局稀疏性


def fuse_conv_and_bn(conv, bn):
    # 融合卷积层和批归一化层 https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备卷积层权重
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备空间偏置
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv  # 返回融合后的卷积层


def model_info(model, verbose=False, img_size=640):
    # 模型信息。img_size 可以是整数或列表，例如 img_size=640 或 img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # 参数总数
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # 梯度参数总数
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # 计算 FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # 输入张量
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # 计算 GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # 如果是整数，扩展为列表
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 的 GFLOPs
    except (ImportError, Exception):
        fs = ''  # 如果出现错误，返回空字符串

    LOGGER.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")  # 输出模型摘要


"""
img: 输入的图像张量，形状为 (batch_size, channels, height, width)。
ratio: 缩放比例，默认值为 1.0，表示不缩放。
same_shape: 布尔值，表示是否保持输入图像的形状，默认为 False。
gs: 网格大小，默认值为 32。
"""
def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # 如果ratio为1.0，即不进行缩放，直接返回原图像。
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]  # 获取输入图像的高度和宽度。
        s = (int(h * ratio), int(w * ratio))  # 计算缩放后的新尺寸s，即新的高度和宽度。
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # 使用双线性插值法 (bilinear) 对图像进行缩放，得到新的图像尺寸。
        if not same_shape:  # 如果 same_shape为 False，则根据网格大小gs计算新的高度和宽度，确保它们是gs的倍数。
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # 对图像进行填充操作，使其尺寸符合新的高度和宽度。填充值为0.447，这是在数据增强时常用的灰色填充值。

def copy_attr(a, b, include=(), exclude=()):
    # 从 b 复制属性到 a，可以选择仅包含 [...] 和排除 [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue  # 如果不在包含列表中，或者是私有属性，或者在排除列表中，则跳过
        else:
            setattr(a, k, v)  # 将属性值设置到 a

class EarlyStopping:
    # 简单的提前停止器
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # 最佳适应度，例如 mAP
        self.best_epoch = 0  # 最佳轮次
        self.patience = patience or float('inf')  # 在适应度停止改善后等待的轮次
        self.possible_stop = False  # 可能在下一个轮次停止
    def __call__(self, epoch, fitness):
        # 调用时检查当前轮次的适应度
        if fitness >= self.best_fitness:  # 允许适应度为零以应对训练初期阶段
            self.best_epoch = epoch  # 更新最佳轮次
            self.best_fitness = fitness  # 更新最佳适应度
        delta = epoch - self.best_epoch  # 无改进的轮次
        self.possible_stop = delta >= (self.patience - 1)  # 下一个轮次可能会停止
        stop = delta >= self.patience  # 如果超出耐心值则停止训练
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop  # 返回是否停止训练

class ModelEMA:
    """
    模型指数移动平均，来源于 https://github.com/rwightman/pytorch-image-models
    保持模型状态字典（参数和缓冲区）中的一切的移动平均。
    这是为了实现类似于
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage 的功能。
    平滑版本的权重对于某些训练方案的良好表现是必要的。
    该类在模型初始化、GPU 分配和分布式训练包装器的顺序中初始化时非常敏感。
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # 创建 EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # EMA 更新次数
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # 指数衰减（帮助早期轮次）
        for p in self.ema.parameters():
            p.requires_grad_(False)  # 不需要梯度
    def update(self, model):
        # 更新 EMA 参数
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)  # 计算衰减值

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # 模型状态字典
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # 仅更新浮点型参数
                    v *= d  # 更新 EMA
                    v += (1 - d) * msd[k].detach()  # 融入当前模型参数
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # 更新 EMA 属性
        copy_attr(self.ema, model, include, exclude)  # 从模型复制属性到 EMA

