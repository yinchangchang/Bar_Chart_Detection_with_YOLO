# coding=utf8
#########################################################################
# File Name: classify_loss.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 14时33分59秒
#########################################################################

import sys
sys.path.append('classification/')

# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# files
import classify_function

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, font_output, font_target, use_weight=None, use_hard_mining=False):
        font_output = self.sigmoid(font_output)
        if use_weight:
            weight = classify_function.get_weight(font_target)
            if self.args.gpu:
                weight = torch.from_numpy(weight).cuda(async=True)
            else:
                weight = torch.from_numpy(weight)
        else:
            weight = None
        font_loss = F.binary_cross_entropy(font_output, font_target, weight)

        # hard_mining 
        if use_hard_mining:
            font_output = font_output.view(-1)
            font_target = font_target.view(-1)
            pos_index = font_target > 0.5
            neg_index = font_target == 0

            # pos
            pos_output = font_output[pos_index]
            pos_target = font_target[pos_index]
            num_hard_pos = max(len(pos_output)/4, min(5, len(pos_output)))
            if len(pos_output) > 5:
                pos_output, pos_target = hard_mining(pos_output, pos_target, num_hard_pos, largest=False)
            pos_loss = self.classify_loss(pos_output, pos_target) * 0.5


            # neg
            num_hard_neg = len(pos_output) * 2
            neg_output = font_output[neg_index]
            neg_target = font_target[neg_index]
            neg_output, neg_target = hard_mining(neg_output, neg_target, num_hard_neg, largest=True)
            neg_loss = self.classify_loss(neg_output, neg_target) * 0.5

            font_loss += pos_loss + neg_loss
            return [font_loss, pos_loss, neg_loss]

        else:
            return [font_loss]


def hard_mining(neg_output, neg_labels, num_hard, largest=True):
    num_hard = min(max(num_hard, 10), len(neg_output))
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)), largest=largest)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels
