# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-anchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import LOGGER, colorstr, emojis

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    # Check anchor order against stride order for  Detect() module m, and correct if necessary
    # 计算每个锚点的面积，并将其展平为一维张量
    a = m.anchors.prod(-1).view(-1)  # anchor area
    # 计算锚点面积的差值
    da = a[-1] - a[0]  # delta a
    # 计算步幅的差值
    ds = m.stride[-1] - m.stride[0]  # delta s
    # 如果锚点面积的顺序与步幅的顺序不一致
    if da.sign() != ds.sign():  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        # 反转锚点的顺序
        m.anchors[:] = m.anchors.flip(0)


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # 检查锚框是否适合数据集，并在必要时重新计算
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # 获取检测模型
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)  # 计算缩放后的形状
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # 随机缩放
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # 获取宽高信息

    def metric(k):  # 计算指标
        r = wh[:, None] / k[None]  # 计算宽高比
        x = torch.min(r, 1 / r).min(2)[0]  # 比率指标
        best = x.max(1)[0]  # 最佳比例
        aat = (x > 1 / thr).float().sum(1).mean()  # 超过阈值的锚框比例
        bpr = (best > 1 / thr).float().mean()  # 最佳可能召回率
        return bpr, aat

    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # 当前锚框
    bpr, aat = metric(anchors.cpu().view(-1, 2))  # 计算当前锚框的BPR和AAD

    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # 如果BPR足够高，说明锚框合适
        LOGGER.info(emojis(f'{s}Current anchors are a good fit to dataset ✅'))
    else:
        LOGGER.info(emojis(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...'))
        na = m.anchors.numel() // 2  # 锚框数量
        try:
            # 尝试通过K均值聚类计算新的锚框
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            LOGGER.info(f'{PREFIX}ERROR: {e}')  # 捕获异常并记录

        new_bpr = metric(anchors)[0]  # 计算新锚框的BPR
        if new_bpr > bpr:  # 如果新锚框更好，则替换
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # 更新锚框
            check_anchor_order(m)  # 检查锚框顺序
            LOGGER.info(f'{PREFIX}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            LOGGER.info(f'{PREFIX}Original anchors better than new anchors. Proceeding with original anchors.')



def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ 创建经过kmeans进化的锚点，从训练数据集中获取

            参数:
                dataset: 数据集路径（data.yaml），或已加载的数据集
                n: 锚点的数量
                img_size: 用于训练的图像尺寸
                thr: 锚点-标签宽高比阈值超参数（用于训练），默认为4.0
                gen: 使用遗传算法进化锚点的代数
                verbose: 是否打印所有结果

            返回:
                k: kmeans进化后的锚点

            使用示例:
                from utils.autoanchor import *; _ = kmean_anchors()
        """
    from scipy.cluster.vq import kmeans  # 导入kmeans函数

    thr = 1 / thr  # 计算阈值的倒数

    def metric(k, wh):  # 计算指标
        r = wh[:, None] / k[None]  # 计算宽高比
        x = torch.min(r, 1 / r).min(2)[0]  # 比率指标
        # x = wh_iou(wh, torch.tensor(k))  # 交并比指标
        return x, x.max(1)[0]  # 返回指标值和最佳指标值

    def anchor_fitness(k):  # 计算适应度
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)  # 计算每个锚点的适应度
        return (best * (best > thr).float()).mean()  # 计算适应度的平均值

    def print_results(k, verbose=True):  # 打印结果
        k = k[np.argsort(k.prod(1))]  # 按锚点面积从小到大排序
        x, best = metric(k, wh0)  # 计算每个锚点的指标值
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # 最佳可能的召回率，超过阈值的锚点数量
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for i, x in enumerate(k):
            s += '%i,%i, ' % (round(x[0]), round(x[1]))  # 添加锚点坐标到结果字符串
        if verbose:
            LOGGER.info(s[:-2])  # 打印结果
        return k

    if isinstance(dataset, str):  # 如果是文件路径
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # 读取数据字典
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)  # 加载数据集

    # 获取标签的宽高
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # 获取所有标签的宽高

    # 过滤掉小于2像素的标签
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f'{PREFIX}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # 过滤掉小于2像素的标签
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans计算
    LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)  # 标准差，用于白化处理
    k, dist = kmeans(wh / s, n, iter=30)  # 运行kmeans算法
    assert len(k) == n, f'{PREFIX}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s  # 恢复原始缩放
    wh = torch.tensor(wh, dtype=torch.float32)  # 转换为张量
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # 转换为张量
    k = print_results(k, verbose=False)  # 打印结果

    # 进化过程
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1   # 适应度、锚点形状、突变概率、标准差
    pbar = tqdm(range(gen), desc=f'{PREFIX}Evolving anchors with Genetic Algorithm:')  # 进度条
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # 突变直到发生变化（防止重复）
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)  # 应用突变
        fg = anchor_fitness(kg)  # 计算突变后的适应度
        if fg > f:   # 如果适应度提高
            f, k = fg, kg.copy()  # 更新适应度和锚点
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'  # 更新进度条描述
            if verbose:
                print_results(k, verbose)  # 打印结果
    return print_results(k)  # 返回最终的锚点
