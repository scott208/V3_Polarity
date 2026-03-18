# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov3.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (copy_attr, fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device,
                               time_sync)

try:
    import thop  # 尝试导入 thop 库，用于计算 FLOPs（浮点运算次数）
except ImportError:
    thop = None  # 如果导入失败，则将 thop 设为 None

class Detect(nn.Module):
    stride = None  # 在构建过程中计算的步幅
    onnx_dynamic = False  # ONNX 导出参数

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # 检测层
        super().__init__()
        self.nc = nc  # 类别数量
        self.no = nc + 5  # 每个锚点的输出数量
        self.nl = len(anchors)  # 检测层数量
        self.na = len(anchors[0]) // 2  # 锚点数量
        self.grid = [torch.zeros(1)] * self.nl  # 初始化网格
        self.anchor_grid = [torch.zeros(1)] * self.nl  # 初始化锚点网格
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # 锚点形状 (nl, na, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # 输出卷积层
        self.inplace = inplace  # 是否使用原地操作（例如切片赋值）

    def forward(self, x):
        z = []  # 推断输出
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # 卷积操作
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) 转换为 x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # 推断阶段
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)  # 生成网格和锚点网格

                y = x[i].sigmoid()  # 应用 Sigmoid 激活函数
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy 位置
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 尺寸
                else:  # 对于 AWS Inferentia 使用的情况
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy 位置
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 尺寸
                    y = torch.cat((xy, wh, y[..., 4:]), -1)  # 拼接
                z.append(y.view(bs, -1, self.no))  # 重塑并添加到结果列表

        return x if self.training else (torch.cat(z, 1), x)  # 返回推断结果

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device  # 获取锚点的设备
        if check_version(torch.__version__, '1.10.0'):  # 检查 PyTorch 版本，兼容旧版本
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()  # 生成网格坐标
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()  # 生成锚点网格
        return grid, anchor_grid

class Model(nn.Module):
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
            Model主要包含模型的搭建与扩展功能，yolov3的作者将这个模块的功能写的很全，
                扩展功能如：特征可视化，打印模型信息、TTA推理增强、融合Conv+Bn加速推理、模型搭载nms功能、autoshape函数：
                模型搭建包含前处理、推理、后处理的模块(预处理 + 推理 + nms)。
            感兴趣的可以仔细看看，不感兴趣的可以直接看__init__和__forward__两个函数即可。

            :params cfg:模型配置文件
            :params ch: input img channels 一般是3 RGB文件
            :params nc: number of classes 数据集的类别个数
            :anchors: 一般是None
        """

        super().__init__()
        # 读取cfg文件中的模型结构配置文件
        if isinstance(cfg, dict):  # 查看cfg是否是字典类型
            self.yaml = cfg   # 如果是字典类型，将cfg赋值给self.yaml
        else:  # 如果cfg不是字典类型，假设cfg是一个YAML文件路径
            import yaml  # 导入yaml库，用于处理YAML文件
            self.yaml_file = Path(cfg).name  # 获取YAML文件的文件名
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='ascii', errors='ignore') as f:   # 以ascii编码方式打开文件，忽略编码错误
                # model dict  取到配置文件中每条的信息（没有注释内容）
                self.yaml = yaml.safe_load(f)  # 使用yaml.safe_load解析文件内容，并赋值给self.yaml

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 获取输入通道数，如果不存在则使用默认值，默认值为3
        if nc and nc != self.yaml['nc']:  # 如果提供的类别数nc与cfg配置字典中的类别数不同，则使用新的nc值。
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")  # 将新值记录到字典中
            self.yaml['nc'] = nc  # 覆盖cfg配置文件中yaml中的nc值
        if anchors:  # 如果提供了锚点值anchors，则使用新的anchors值，并记录日志信息。
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # 覆盖 cfg配置文件中yaml中的anchor值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # 调用parse_model函数解析模型结构，传入深拷贝的配置字典和输入通道数，生成模型和保存列表。
        self.names = [str(i) for i in range(self.yaml['nc'])]  # 根据类别数生成默认的类别名称列表，例如['0', '1', '2', ...]。
        self.inplace = self.yaml.get('inplace', True)  # 从配置字典中获取inplace值，如果不存在则默认设为True。

        # 构建步长和锚点
        m = self.model[-1]  # 获取模型的最后一层，通常是Detect层
        if isinstance(m, Detect):  # 如果最后一层为Detect则执行如下代码
            s = 256  # 2x min stride  # 设置输入图像的尺寸（2倍的最小步长）
            m.inplace = self.inplace

            # # 假设640X640的图片大小，在最后三层时分别乘1/8 1/16 1/32，得到80，40，20，这个stride是模型的下采样的倍数
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)  # 将当前图片的大小处理成相对当前feature map的anchor大小如[10, 13]/8 -> [1.25, 1.625]
            check_anchor_order(m)  # 检查anchor顺序与stride顺序是否一致
            self.stride = m.stride  # 保存步长到self.stride。
            self._initialize_biases()  # 初始化偏置（仅执行一次）

        # 初始化权重和偏置
        initialize_weights(self)  # 初始化模型的权重
        self.info()  # 打印模型信息
        LOGGER.info('')  # 记录日志信息

    #  定义了一个forward方法，这个方法根据传入的参数选择不同的前向推断方式
    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # 调用self._forward_augment(x) 方法进行增强推断
        return self._forward_once(x, profile, visualize)  #  如果 augment参数为False，方法进行单尺度推断

    # 个方法通过不同的尺度和翻转方式对输入数据 x 进行增强推断，返回增强推断的结果。
    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # 获取输入数据 x 的图像尺寸，假设是高度和宽度。
        s = [1, 0.83, 0.67]  # scales，尺度列表，用于对输入图像进行缩放
        f = [None, 3, None]  # flips (2-ud, 3-lr)，翻转方式列表，用于对输入图像进行翻转操作（2-上下翻转，3-左右翻转）。
        y = []  # 存储推断结果的列表。
        for si, fi in zip(s, f):  # 遍历尺度s和翻转方式f
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max())) # 对图像进行缩放和翻转。
            yi = self._forward_once(xi)[0]  # forward，将增强后的图像输入模型进行一次前向传播。
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)  # 将预测结果反缩放到原始图像尺寸。
            y.append(yi)   # 将处理后的预测结果添加到结果列表y中
        y = self._clip_augmented(y)  # 剪切增强后的结果（如果需要）。
        return torch.cat(y, 1), None  # 将处理后的预测结果添加到结果列表y中。

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # 初始化 y（存储各层输出的列表）和 dt（存储各层运行时间的列表，用于性能分析）。
        for m in self.model:  # 遍历模型中的每一层m。
            if m.f != -1:  # 检查当前层的输入是否来自上一层。
                # 如果 m.f 是整数，则从输出列表 y 中获取第 m.f 层的输出作为输入 x。
                # 如果 m.f 是列表，则根据列表中的索引从 y 中获取对应的输出。
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 启用了性能分析（profile=True）
            if profile:
                self._profile_one_layer(m, x, dt)
            # 将输入x传递给当前层m，执行前向传播。
            x = m(x)
            y.append(x if m.i in self.save else None)  # 将当前层的输出x添加到列表y中。如果当前层的索 m.i在需要保存的输出层索引列表self.save 中，则保存输出；否则保存 None。
            # 启用了特征可视化（visualize=True），
            # 则调用 feature_visualization 方法对当前层的输出进行可视化，并保存到指定目录。
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    # 用于对经过增强推理后的预测结果进行反操作（去尺度、去翻转），以恢复到原始图像的尺度和方向。
    def _descale_pred(self, p, flips, scale, img_size):
        """
        p: 预测结果张量。
        flips: 翻转方式（2表示上下翻转，3表示左右翻转）。
        scale: 缩放比例。
        img_size: 原始图像的尺寸（高度和宽度）。
        """
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # 如果self.inplace为True，直接对预测结果进行去尺度操作，缩放p的前四个元素（坐标和宽高）。
            # 根据翻转方式，进行相应的反翻转操作。
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        #  如果self.inplace为False，分别对p的前四个元素进行去尺度操作。
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            #  根据翻转方式，进行相应的反翻转操作。
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)  # 将处理后的 x、y、宽高和剩余预测结果拼接成新的预测结果张量。
        return p

    def _clip_augmented(self, y):
        # Clip  augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y


    # 用于分析单个层的计算性能，包括计算 FLOPs（浮点运算数）和执行时间。
    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # 用于检查当前层m是否是Detect 层，如果是，则设置c为True。
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # 使用 thop.profile 方法计算当前层的 FLOPs。
        t = time_sync()  # 记录当前时间t。
        for _ in range(10):  # 执行当前层的前向传播10次，并记录执行时间。
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)  # 将平均执行时间（乘以100以转换为毫秒）添加到dt列表中。
        # 如果当前层是模型的第一层，输出标题行。
        # 输出当前层的分析结果，包括执行时间、GFLOPs、参数数量和层类型。
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        # 如果当前层是 Detect 层，输出总执行时间。
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # 它通过计算和设置偏置项的初始值，以便更好地训练模型。
    # 可选参数 cf 表示类别频率，如果提供，将用于调整类别预测的偏置项。
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # 获取 Detect 模块
        m = self.model[-1]  # Detect() module
        # 遍历Detect模块的每个子模块和对应的步幅
        for mi, s in zip(m.m, m.stride):  # from
            # 将偏置项mi.bias重塑为形状(m.na, -1)，其中m.na是锚点的数量。
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # 为对象置信度（第4个位置）添加偏置项。
            # 计算基于每个步幅 s 的初始值，假设每张640x640的图像有8个对象。
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)

            # 为类别预测（从第5个位置开始）添加偏置项。
            # 如果没有提供cf，则使用默认值；否则，根据cf计算偏置项。
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            # 将调整后的偏置项b重新赋值给mi.bias，并确保其在训练过程中可学习
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # 获取最后一个模型模块，即 Detect() 模块
        for mi in m.m:  # 遍历所有卷积层
            b = mi.bias.detach().view(m.na, -1).T  # 获取偏置，将其从 (255) 转换为 (3, 85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
            # 打印卷积层的偏置信息，包括卷积核的数量和每个偏置的均值

    def fuse(self):  # 融合模型中的 Conv2d() 和 BatchNorm2d() 层
        LOGGER.info('正在融合层...')
        for m in self.model.modules():
            # 如果模块是 Conv 类型且具有 'bn' 属性
            if isinstance(m, Conv) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 融合卷积层和批归一化层，更新卷积层
                delattr(m, 'bn')  # 删除批归一化层属性
                m.forward = m.forward_fuse  # 更新前向传播函数为融合后的版本
        self.info()  # 打印模型信息
        return self  # 返回当前对象

    def autoshape(self):  # 添加 AutoShape 模块
        LOGGER.info('正在添加 AutoShape...')
        m = AutoShape(self)  # 包装模型为 AutoShape 模型
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # 复制属性
        return m  # 返回 AutoShape 模型

    def info(self, verbose=False, img_size=640):  # 打印模型信息
        model_info(self, verbose, img_size)  # 调用 model_info 函数来打印模型信息

    def _apply(self, fn):
        # 将 to()、cpu()、cuda()、half() 应用到模型中不是参数或已注册的缓冲区的张量
        self = super()._apply(fn)  # 调用父类的 _apply 方法
        m = self.model[-1]  # 获取模型的最后一层，即 Detect() 层
        if isinstance(m, Detect):  # 如果最后一层是 Detect 实例
            m.stride = fn(m.stride)  # 应用 fn 函数到 stride 张量
            m.grid = list(map(fn, m.grid))  # 应用 fn 函数到 grid 张量列表
            if isinstance(m.anchor_grid, list):  # 如果 anchor_grid 是列表
                m.anchor_grid = list(map(fn, m.anchor_grid))  # 应用 fn 函数到 anchor_grid 张量列表
        return self  # 返回处理后的模型

def parse_model(d, ch): # 将解析的网络模型结构作为输入，是字典形式，输入通道数（通常为3）
    """
        主要功能：parse_model模块用来解析模型文件(从Model中传来的字典形式)，并搭建网络结构。
        在上面Model模块的__init__函数中调用

        这个函数其实主要做的就是: 更新当前层的args（参数）,计算c2（当前层的输出channel） =>
                              使用当前层的参数搭建当前层 =>
                              生成 layers + save

        :params d: model_dict 模型文件 字典形式 yolov3.yaml中的网络结构元素 + ch
        :params ch: 记录模型每一层的输出channel 初始ch=[3] 后面会删除
        :return nn.Sequential(*layers): 网络的每一层的层结构
        :return sorted(save): 把所有层结构中from不是-1的值记下 并排序 [4, 6, 10, 14, 17, 20, 23]
    """
    # LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")

    # 读取d字典中的anchors和parameters(nc、depth_multiple、width_multiple)
    #  nc（number of classes）数据集类别个数；
    # depth_multiple，通过深度参数depth gain在搭建每一层的时候，实际深度 = 理论深度(每一层的参数n) * depth_multiple，起到一个动态调整模型深度的作用。
    # width_multiple，在模型中间层的每一层的实际输出channel = 理论channel(每一层的参数c2) * width_multiple，起到一个动态调整模型宽度的作用。
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']

    """
    如果anchors是一个列表，则计算锚点的数量na。
    具体来说，取anchors列表的第一个元素的长度除以2，
    因为每个锚点由两个值（宽度和高度）表示。如果anchors不是列表，则直接使用 anchors 的值。
    """
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # na为每个检测头的anchor数量
    no = na * (nc + 5)  # 计算输出数量 no。输出数量等于锚点数量乘以（类别数nc加上5）。这里的5包括4个边界框坐标（x, y, w, h）和1个置信度分数。

    # 开始搭建网络
    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel
    layers, save, c2 = [], [], ch[-1]  # layers: 保存每一层的层结构，save: 记录下所有层结构中from中不是-1的层结构序号，c2: 保存当前层的输出channel
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # 遍历模型的backbone和head部分，获取from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # 如果m是字符串，则使用eval函数将其转换为实际的模块类或函数，计算该模块的值。
        for j, a in enumerate(args):
            try:
                # 如果 a 是一个字符串，则使用 eval(a) 计算其值，并将结果赋给 args[j]
                # 如果 a 不是字符串，则直接将 a 赋给 args[j]。
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass
        # print("argshaha", args)

        # 该部分借用yolov5算法的中方法，利用调整系数gd来改变对应模块的重复次数，以达到增大模型大小的目标
        # 原本的yolov3是没有这个功能的，该版本的代码传承了Ultralytics公司的，yolov5就是该公司出品的
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
        #          BottleneckCSP, C3, C3TR, C3SPP, C3Ghost]:

        if m in [Conv,  Bottleneck,  SPP,  MixConv2d, Focus, CrossConv]:
            c1, c2 = ch[f], args[0]  # 获取当前层输入通道数 c1 和输出通道数 c2。
            if c2 != no:  # if not output # 判断是否等于输出通道大小。
                # make_divisible 函数的作用是将输入x调整为大于或等于x且能被divisor整除的最小整数。
                # 它使用 math.ceil 函数来实现这一目的。
                # 其中调整系数gw来改变对应模块的通道大小
                # 原本的yolov3是没有这个功能的，该版本的代码传承了Ultralytics公司的，yolov5就是该公司出品的
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]  # 更新 args，将输入通道数 c1 和调整后的输出通道数 c2 作为新的参数列表的前两个元素。
        elif m is nn.BatchNorm2d:
            args = [ch[f]]  # 仅将输入通道数 ch[f] 作为参数 args。
        elif m is Concat:
            c2 = sum(ch[x] for x in f)  # 计算多个输入通道数 ch[x] 的总和，得到新的输出通道数 c2。
        elif m is Detect:
            args.append([ch[x] for x in f]) # 在参数 args 中附加包含输入通道数的列表 ch[x]。
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)  #如果 args[1] 是整数，则将其转换为包含适当数量的锚框数的列表。
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2  # 根据输入通道数 ch[f] 和参数 args[0] 的平方，计算新的输出通道数 c2。
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2  # 根据输入通道数 ch[f] 和参数 args[0] 的平方，计算新的输出通道数 c2
        else:
            c2 = ch[f]  # 其他的情况，默认将当前输入通道数 ch[f] 作为输出通道数 c2

        # 在Python中，前面的*符号用于解包参数列表。*args 允许你将一个参数列表传递给函数
        # 而在函数内部可以将这个参数列表解包成单独的参数。
        # 义了一个变量m_，其值取决于变量n的大小。如果n大于1，则创建一个包含n个m(*args)实例的nn.Sequential模块；
        # 否则，直接创建一个 m(*args) 实例。具体来说，这段代码是在处理神经网络模块的堆叠和实例化。
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 这行代码将模块m转换为字符串，并截取其类型字符串的中间部分（去掉前8个字符和最后2个字符），然后去掉__main__.前缀。
        t = str(m)[8:-2].replace('__main__.', '')  # module type

        # 这行代码计算模块m_ 中所有参数的总数量。m_.parameters()
        # 返回模块的参数迭代器，x.numel() 返回每个参数的元素数量，sum 计算所有参数的总数量。
        np = sum(x.numel() for x in m_.parameters())  # number params
        # 这行代码将索引i、'from'索引f、模块类型字符串t、参数数量np附加到模块m_上，方便后续使用。
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # 将满足条件的元素添加到 save 列表中
        # 将模块 m_ 添加到 layers 列表中。
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)

        # 初始化列表ch，并不断保存输出通道数到该列表中。
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3.yaml', help='model.yaml')  # 添加 --cfg 参数，类型为字符串，默认值为 yolov3.yaml，用于指定模型配置文件。
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # 添加 --device 参数，默认值为空字符串，用于指定要使用的 CUDA 设备或 CPU。
    parser.add_argument('--profile', action='store_true', help='profile model speed')  # 添加 --profile参数，类型为布尔值，如果在命令行中包含该参数，则opt.profile将为True，用于启用模型速度分析
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')  # 添加--test参数，类型为布尔值，如果在命令行中包含该参数，则opt.test将为True，用于测试所有yolo配置文件。
    opt = parser.parse_args()  # 解析命令行参数并将结果存储在 opt 对象中。

    opt.cfg = check_yaml(opt.cfg)  # 检查文件格式是否正确
    print_args(FILE.stem, opt)  # 打印参数

    device = select_device(opt.device)  # 利用select_device获取设备信息

    # 创建一个模型实例，将其移动到指定的设备（CPU 或 GPU），并将模型设置为训练模式
    model = Model(opt.cfg).to(device)
    model.train()

    # 用来判断是否启用了 --profile 选项。
    # 如果启用，它将创建一个随机生成的图像张量并将其传递给模型进行推理，同时记录模型性能（例如速度）
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # test选项。如果启用，它将遍历指定目录中的所有符合特定模式的配置文件，
    # 并尝试使用这些配置文件来实例化模型。如果过程中发生异常，将打印出错误信息。
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')