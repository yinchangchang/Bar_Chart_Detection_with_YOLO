# coding=utf8
#########################################################################
# File Name: classify_function.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 14时34分13秒
#########################################################################


import numpy as np


# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

def get_weight(labels):
    '''
    多标签分类情况下计算权重
    '''
    labels = labels.data.cpu().numpy()

    # 按每个维度计算weight
    weights = np.zeros_like(labels)
    weight_false = 1.0 / ((labels==0).sum(0) + 10e-20)
    label_true = (labels==1).sum(0)
    for i in range(labels.shape[1]):    # 为每一维计算权重
        label_i = labels[:,i]
        weight_i = np.ones(labels.shape[0]) * weight_false[i]   # false权重加和为1
        if label_true[i] > 0:
            weight_i[label_i==1] = 1.0 / label_true[i]          # true权重加和为1
        weights[:,i] = weight_i
    weights *= np.ones_like(labels).sum() / (weights.sum() + 10e-20)
    weights[labels<-0.5] = 0

    '''
    weight_dim = weights
    # 按每条数据计算weight
    weights = np.zeros_like(labels)
    weight_false = 1.0 / ((labels==0).sum(1) + 10e-20)
    label_true = (labels==1).sum(1)
    for i in range(labels.shape[0]):    # 为每条数据计算权重
        label_i = labels[i, :]
        weight_i = np.ones(labels.shape[1]) * weight_false[i]   # false权重加和为1
        if label_true[i] > 0:
            weight_i[label_i==1] = 0.05 / label_true[i]          # true权重加和为1
        weights[i, :] = weight_i
    weights *= np.ones_like(labels).sum() / (weights.sum() + 10e-20)
    weights[labels<-0.5] = 0
    weights = (weights + weight_dim) / 2
    # '''

    return weights
