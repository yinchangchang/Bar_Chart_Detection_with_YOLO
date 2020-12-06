# coding=utf8
#########################################################################
# File Name: classify_output.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 14时33分52秒
#########################################################################

# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

class classify_fc(nn.Module):
    '''
    使用全局特征分类
    '''

class classify_conv(nn.Module):
    '''
    使用局部特征分类
    '''
    def __init__(self, in_size, out_size):
        super(classify_conv, self).__init__()
        self.inplanes = in_size
        num_ftrs = in_size
        self.classifier= nn.Sequential(
                nn.Conv2d(num_ftrs, num_ftrs, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_ftrs, out_size, kernel_size=1, bias=False)
        )
        self.train_params = []
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, feature, phase='train'):
        out = self.classifier(feature)
        out_size = out.size()
        out = out.view(out.size(0),out.size(1),-1)
        if phase == 'train':
            all_out = out
            out = F.adaptive_max_pool1d(out, output_size=(1)).view(out.size(0),-1) # (32, 1824)
            return out, all_out
        else:
            out = out.transpose(1,2).contiguous()
            out = out.view(out_size[0],out_size[2], out_size[3], out_size[1]) # (32, 1, 8, 1824)
            return out, features
