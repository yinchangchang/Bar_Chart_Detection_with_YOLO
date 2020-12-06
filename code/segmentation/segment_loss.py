# coding=utf8
#########################################################################
# File Name: segment_loss.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月13日 星期四 10时16分34秒
#########################################################################

import sys
sys.path.append('segmentation/')

# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# files
import segment_function

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.m_loss = nn.CrossEntropyLoss()
        self.regress_loss = nn.SmoothL1Loss()

    def transpose(self, out):
        # 改变格式
        size = out.size()
        out = out.view(size[0], size[1], size[2]*size[3])
        out = out.transpose(1,2).contiguous()
        out = out.view(size[0], size[2], size[3], size[1])
        return out

    def forward(self, output, target, use_weight=None, use_hard_mining=False):
        '''
        多标签分割
        output: [bs, 1 + n_c, w, h]
        target: [bs, 2, w, h]
        '''

        # transpose
        output = self.transpose(output)
        target = self.transpose(target)

        # reshape
        output = output.view(-1, output.size(-1))
        target = target.view(-1, target.size(-1))

        # 区分二分类和多分类
        output_m = output[:, 1:]
        target_m = target[:, 1]
        output = output[:, 0]
        target = target[:, 0]

        # 二分类
        b_loss, b_loss_data = 0,0
        pos_idcs = target == 1
        pos_output = output[pos_idcs]
        pos_target = target[pos_idcs]
        if len(pos_output):
            pos_loss = self.classify_loss(self.sigmoid(pos_output), pos_target)
            b_loss += pos_loss
            b_loss_data += pos_loss.data.cpu()
        pos_correct = (pos_output.data.cpu() > 0).sum()

        neg_idcs = target == 0
        neg_output = output[neg_idcs]
        neg_target = target[neg_idcs]
        if len(neg_output):
            neg_loss = self.classify_loss(self.sigmoid(neg_output), neg_target)
            b_loss += neg_loss
            b_loss_data += neg_loss.data.cpu()
        neg_correct = (neg_output.data.cpu() < 0).sum()

        # 多分类
        pos_idcs_m = target_m > -0.5
        output_m, target_m = segment_function.select(output_m, target_m, pos_idcs_m.float())
        target_m = target_m.long()
        if len(output_m):
            m_loss = self.m_loss(output_m, target_m)
            output_no, output_id = torch.max(output_m, 1)
            correct_m = (output_id == target_m).sum().data.cpu().numpy()
            m_loss_data = m_loss.data.cpu()
        else:
            m_loss, m_loss_data, correct_m = 0, 0, 0

        loss = b_loss + m_loss

        return [loss, b_loss_data, pos_correct, len(pos_output), neg_correct, len(neg_output)] + [m_loss_data, correct_m, len(output_m) ]



def hard_mining(neg_output, neg_labels, num_hard, largest=True):
    num_hard = min(max(num_hard, 10), len(neg_output))
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)), largest=largest)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels
