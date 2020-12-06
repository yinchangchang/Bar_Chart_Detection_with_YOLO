# coding=utf8
#########################################################################
# File Name: segment_line.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 11时42分30秒
#########################################################################


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


class segment_line(nn.Module):
    '''
    使用局部特征对有字的区域进行分割
    '''
    def __init__(self, in_size, out_size):
        super(segment_line, self).__init__()
        self.inplanes = in_size
        num_ftrs = in_size
        self.classifier= nn.Sequential(
                nn.Conv2d(num_ftrs, num_ftrs, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs, out_size, kernel_size=1, bias=False)
        )
        self.train_params = []
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, feature):
        out = self.classifier(feature)
        return out
