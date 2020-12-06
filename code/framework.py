# coding=utf8
#########################################################################
# File Name: framework.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 11时22分44秒
#########################################################################


# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

'''
framework.py 包含整个网络架构
'''

import models
from models import densenet,resnet
import classification, segmentation, detection
from classification import classify_output, classify_loss
from segmentation import segment_line, segment_loss
from detection import detect_character, detect_loss

def build_model(p_dict):

    args = p_dict['args']                   # 全局参数
    num_class = p_dict['num_classes']       # 类别

    model_dict = dict()
    loss_dict = dict()

    ### 构建模型
    if args.model == 'densenet':
        if args.gpu:
            # feature_model = densenet.densenet121().cuda()
            feature_model = densenet.DenseNet().cuda()
        else:
            # feature_model = densenet.densenet121()
            feature_model = densenet.DenseNet()
    elif args.model == 'resnet':
        if args.gpu:
            feature_model = resenet.ResNet().cuda()
        else:
            feature_model = resenet.ResNet()
    elif args.model == 'unet':
        if args.gpu:
            feature_model = unet.UNet().cuda()
        else:
            feature_model = unet.UNet()
    model_dict['feature'] = feature_model

    if args.gpu:
        # model_dict['classify'] = classify_output.classify_conv(1024, num_class).cuda()
        # model_dict['segment_line'] = segment_line.segment_line(1024, 1 + num_class).cuda()
        model_dict['detect_character'] = detect_character.detect(256, (5 + num_class) * len(args.anchors), len(args.anchors)).cuda()
    else:
        # model_dict['classify'] = classify_output.classify_conv(1024, num_class)
        # model_dict['segment_line'] = segment_line.segment_line(1024, 1 + num_class)
        model_dict['detect_character'] = detect_character.detect(256, (5 + num_class) * len(args.anchors), len(args.anchors))

    ### 构建loss函数
    if args.gpu:
        # loss_dict['classify'] = classify_loss.Loss(args).cuda()
        loss_dict['segment_line'] = segment_loss.Loss().cuda()
        loss_dict['detect_character'] = detect_loss.Loss().cuda()
    else:
        # loss_dict['classify'] = classify_loss.Loss(args)
        loss_dict['segment_line'] = segment_loss.Loss()
        loss_dict['detect_character'] = detect_loss.Loss()

    ### 并发
    # if 0 and args.use_parallel:
    # if 1 and :
    if args.gpu:
        for k,v in model_dict.items():
            model_dict[k] = torch.nn.DataParallel(v)
        # for k,v in loss_dict.items():
        #     loss_dict[k] = torch.nn.DataParallel(v)

    p_dict['model_dict'] = model_dict
    p_dict['loss_dict'] = loss_dict
