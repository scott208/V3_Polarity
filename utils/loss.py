# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):
    # 返回平滑的正负标签，用于二元交叉熵损失计算
    # 参数：
    #   eps: 标签平滑的比例，默认值为 0.1
    # 返回：
    #   positive: 平滑后的正标签值
    #   negative: 平滑后的负标签值
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # 使用改进的 BCEWithLogitsLoss，减少缺失标签的影响
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # 必须使用 nn.BCEWithLogitsLoss()
        self.alpha = alpha  # 控制平滑程度的超参数

    def forward(self, pred, true):
        # 计算损失
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # 将 logits 转换为概率
        dx = pred - true  # 计算预测与真实标签之间的差异
        # dx = (pred - true).abs()  # 可选：考虑缺失标签和错误标签的影响
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))  # 计算平滑因子
        loss *= alpha_factor  # 应用平滑因子
        return loss.mean()  # 返回平均损失


class FocalLoss(nn.Module):
    # 将焦点损失包装在现有损失函数中，例如：criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # 必须是 nn.BCEWithLogitsLoss()
        self.gamma = gamma  # 调整因子
        self.alpha = alpha  # 平衡因子
        self.reduction = loss_fcn.reduction  # 保存原始的 reduction 设置
        self.loss_fcn.reduction = 'none'  # 需要对每个元素应用焦点损失

    def forward(self, pred, true):
        # 计算基础损失
        loss = self.loss_fcn(pred, true)

        # 计算预测概率
        pred_prob = torch.sigmoid(pred)  # 从 logits 转换为概率
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # 计算 p_t
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 计算平衡因子
        modulating_factor = (1.0 - p_t) ** self.gamma  # 计算调制因子
        loss *= alpha_factor * modulating_factor  # 应用焦点损失调整

        # 根据原始的 reduction 设置返回损失
        if self.reduction == 'mean':
            return loss.mean()  # 返回平均损失
        elif self.reduction == 'sum':
            return loss.sum()  # 返回总损失
        else:  # 'none'
            return loss  # 返回原始损失


class QFocalLoss(nn.Module):
    # 将质量焦点损失包装在现有损失函数中，例如：criteria = QFocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # 必须是 nn.BCEWithLogitsLoss()
        self.gamma = gamma  # 调整因子
        self.alpha = alpha  # 平衡因子
        self.reduction = loss_fcn.reduction  # 保存原始的 reduction 设置
        self.loss_fcn.reduction = 'none'  # 需要对每个元素应用焦点损失

    def forward(self, pred, true):
        # 计算基础损失
        loss = self.loss_fcn(pred, true)
        # 计算预测概率
        pred_prob = torch.sigmoid(pred)  # 从 logits 转换为概率
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 计算平衡因子
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma  # 计算调制因子
        loss *= alpha_factor * modulating_factor  # 应用质量焦点损失调整

        # 根据原始的 reduction 设置返回损失
        if self.reduction == 'mean':
            return loss.mean()  # 返回平均损失
        elif self.reduction == 'sum':
            return loss.sum()  # 返回总损失
        else:  # 'none'
            return loss  # 返回原始损失


class ComputeLoss:
    # 计算损失
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # 获取模型设备
        h = model.hyp  # 超参数

        # 定义损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # 类别标签平滑 https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # 正负 BCE 目标

        # 焦点损失
        g = h['fl_gamma']  # 焦点损失的 gamma 值
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() 模块
        # 设定平衡因子
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 的索引
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))  # 设置属性

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # 获取目标

        # 计算损失
        for i, pi in enumerate(p):  # 层索引，层预测
            b, a, gj, gi = indices[i]  # 图像、锚点、网格y、网格x
            tobj = torch.zeros_like(pi[..., 0], device=device)  # 目标对象

            n = b.shape[0]  # 目标数量
            if n:
                ps = pi[b, a, gj, gi]  # 与目标对应的预测子集

                # 回归损失
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 预测框
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # 计算 IoU
                lbox += (1.0 - iou).mean()  # IoU 损失

                # 对象性损失
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # IoU 比例

                # 分类损失
                if self.nc > 1:  # 仅在有多个类别时计算分类损失
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # 目标
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # 计算 BCE 损失

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # 对象性损失
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # 批大小

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # 为 compute_loss() 构建目标，输入目标为 (image, class, x, y, w, h)
        na, nt = self.na, targets.shape[0]  # 锚点数量，目标数量
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # 归一化到网格空间的增益
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # 与 nt 重复相同
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # 添加锚点索引

        g = 0.5  # 偏差
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j, k, l, m
                            ], device=targets.device).float() * g  # 偏移量

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy 增益

            # 将目标与锚点匹配
            t = targets * gain
            if nt:
                # 匹配
                r = t[:, :, 4:6] / anchors[:, None]  # 宽高比
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # 比较
                t = t[j]  # 过滤

                # 偏移量
                gxy = t[:, 2:4]  # 网格 xy
                gxi = gain[[2, 3]] - gxy  # 逆偏移
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # 定义
            b, c = t[:, :2].long().T  # 图像，类别
            gxy = t[:, 2:4]  # 网格 xy
            gwh = t[:, 4:6]  # 网格宽高
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # 网格 xy 索引

            # 添加
            a = t[:, 6].long()  # 锚点索引
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # 图像，锚点，网格索引
            # indices.append((b, a, gj.clamp_(0, gain[3] - 1).long(), gi.clamp_(0, gain[2] - 1).long()))  # 图像，锚点，网格索引
            # 修改为以下分步处理
            gj = torch.clamp(gj, 0, gain[3]-1).round().to(device=targets.device, dtype=torch.int64)
            gi = torch.clamp(gi, 0, gain[2]-1).round().to(device=targets.device, dtype=torch.int64)
            indices.append((b, a, gj, gi))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # 盒子
            anch.append(anchors[a])  # 锚点
            tcls.append(c)  # 类别

        return tcls, tbox, indices, anch