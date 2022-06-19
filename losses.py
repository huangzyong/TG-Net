import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.001

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class BinaryTverskyLossV2(nn.Module):

    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean'):
        """Dice loss of binary class
        Args:
            alpha: controls the penalty for false positives.
            beta: penalty for false negative. Larger beta weigh recall higher
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        Shapes:
            output: A tensor of shape [N, 1,(d,) h, w] without sigmoid activation function applied
            target: A tensor of shape same with output
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """
        super(BinaryTverskyLossV2, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = 10
        self.reduction = reduction
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target, mask=None):
        batch_size = output.size(0)
        bg_target = 1 - target
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)  # can not use inplace for bp
            target = target.float().mul(valid_mask)
            bg_target = bg_target.float().mul(valid_mask)

        output = torch.sigmoid(output).view(batch_size, -1)
        target = target.view(batch_size, -1)
        bg_target = bg_target.view(batch_size, -1)

        P_G = torch.sum(output * target, 1)  # TP
        P_NG = torch.sum(output * bg_target, 1)  # FP
        NP_G = torch.sum((1 - output) * target, 1)  # FN

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)

        loss = 1. - tversky_index
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.99, num=0):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)
        self.BinaryTverskyLoss = BinaryTverskyLossV2()
        self.ce = nn.CrossEntropyLoss()
        self.delay = alpha ** num

    def forward(self, input, target):
        if self.delay == 1:
            loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        else:
            loss = self.delay * self.dice_loss(input, target) + (1 - self.delay) * self.focal_loss(input, target)
        return loss


def cos_loss(prediction, target):
    """
    Calculate the cosine similarity between prediction and target
    Input is 3D
    First, convert 3D to 2D
    Representing the results as multiple slices
    inputs_shape: [batch_size, channel, x, y, z]
    target_shape: [batch_size, channel, x, y, z]
    input_type: tensor
    output_type: tensor
    output: loss value
    """
    loss = []
    for b in range(target.shape[0]):
        cos_total = 0
        for s in range(target.shape[-1]):
            pred = prediction[b, 0, :, :, s]  # shape: [x, y]
            gt = target[b, 0, :, :, s]
            pred = pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()

            # pred_norm = F.normalize(pred, p=2, dim=0)  # p=2 表示求二范数，dim=2 表示按列标准化（单位化））
            # gt_norm = F.normalize(gt, p=2, dim=0)  # 如果某一列全为0，标准化（单位化）之后仍然0
            # gt_norm = gt_norm  # 转置
            # res = np.dot(pred_norm.T, gt_norm)  # TODO 求每一列的余弦相似度 ？？？？？？？
            # diag = np.diagonal(res)

            re = np.dot(pred.T, gt)  # 向量点乘
            x = np.linalg.norm(pred, axis=0)
            y = np.linalg.norm(gt, axis=0).reshape(-1, 1)
            x[x == 0] = 1
            y[y == 0] = 1
            mul = x * y
            # nom = np.linalg.norm(pred, axis=0) * np.linalg.norm(gt, axis=0).reshape(-1, 1)  # 求模长的乘积
            res = re / mul
            diag = np.diagonal(res) * 0.5 + 0.5  # 提取每个列向量得余弦相似度并将[-1,1]线性映射为[0,1]
            cos_total += diag.mean()
        cos_s = cos_total / target.shape[-1]
        loss.append(1-cos_s)
    return np.mean(loss)


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]
        i0 = 1
        i1 = 2
        while i1 < len(inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(inp, target)