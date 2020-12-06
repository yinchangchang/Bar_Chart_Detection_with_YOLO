# coding=utf8
#########################################################################
# File Name: detect_character.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 11时42分13秒
#########################################################################


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


class detect(nn.Module):
    '''
    使用局部特征对单个字进行检测
    '''
    def __init__(self, in_size, out_size, n_anchors):
        super(detect, self).__init__()
        self.inplanes = in_size
        num_ftrs = in_size
        self.detect = nn.Sequential(
                nn.Conv2d(num_ftrs, num_ftrs, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs, out_size, kernel_size=1, bias=False)
        )
        self.train_params = []
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.n_anchors = n_anchors

    def forward(self, feature):
        # print(feature.size()) 
        out = self.detect(feature)
        # print(out.size()) 

        # 改变格式
        size = out.size()
        out = out.view(size[0], size[1], size[2]*size[3])
        out = out.transpose(1,2).contiguous()
        out = out.view(size[0], size[2], size[3], self.n_anchors, int(size[1]/self.n_anchors))
        return out
