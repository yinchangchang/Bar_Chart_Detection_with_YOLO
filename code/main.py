# coding=utf8


'''
main.py 为程序入口
'''


# 基本依赖包
import os
import sys
import time
import json
import traceback
import numpy as np
from glob import glob
from tqdm import tqdm
from tools import parse


# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


# 自定义文件
import function
import loaddata
import framework
from loaddata import dataloader


# 全局变量
args = parse.args
args.stride = 4
args.use_segmention = 0
args.use_detection = 1
args.gpu = 1

args.anchors = []
args.input_shape = [512, 512]
# args.input_shape = [128, 128]

# text
x_delta_list = [4, 8, 16, 32, 48, 64]
y_delta_list = [3, 6, 12, 24]
for x in x_delta_list:
    for y in y_delta_list:
        args.anchors.append([x, y])
        args.anchors.append([y, x])
# bar
x_delta_list = [40, 80, 120, 180, 240, 320, 500]
y_delta_list = [32, 64, 96, 128, 160]
for x in x_delta_list:
    for y in y_delta_list:
        args.anchors.append([x, y])
        args.anchors.append([y, x])

print('There are {:d} anchors.'.format(len(args.anchors)))

xya = []
for ix in range(int(args.input_shape[0] / args.stride)):
    for iy in range(int(args.input_shape[1] / args.stride)):
        for ia in range(len(args.anchors)):
            xya.append([ix,iy,ia])
args.xya = np.array(xya)

def test(p_dict, have_label=0):
    ### 传入参数
    epoch = p_dict['epoch']
    data_loader= p_dict['test_loader']            # 测试数据
    model_dict = p_dict['model_dict']           # 模型
    if have_label:
        loss_dict = p_dict['loss_dict']             # loss 函数

    ### 局部变量定义
    classification_metric_dict = dict()
    segmentation_metric_dict = dict()
    detection_metric_dict = dict()

    ### 模型训练
    for i,data in enumerate(tqdm(data_loader)):

        # images, labels = [Variable(x.cuda(async=True)) for x in data[1:3]]
        if args.gpu:
            images = Variable(data[1].cuda(async=True))
        else:
            images = Variable(data[1])

        # 提取特征
        features = model_dict['feature'](images)

        loss_gradient = 0

        ## 分类
        # 计算输出以及loss
        '''
        outputs, all_outputs = model_dict['classify'](features)
        if have_label:
            if args.gpu:
                labels = Variable(data[2].cuda(async=True))
            else:
                labels = Variable(data[2])
            classification_loss_output = loss_dict['classify'](outputs, labels, 1)                 # 分类loss
            loss_gradient += classification_loss_output[0]
            # 计算性能指标
            function.compute_metric(outputs, labels, classification_loss_output, classification_metric_dict)
        '''

        ## 分割
        if args.use_segmention:
            # 计算输出以及loss
            segment_line_output = model_dict['segment_line'](features)
            if have_label:
                if args.gpu:
                    segment_line_labels = Variable(data[3].cuda(async=True))
                else:
                    segment_line_labels = Variable(data[3])
                segment_line_loss_output = loss_dict['segment_line'](segment_line_output, segment_line_labels)
                loss_gradient += segment_line_loss_output[0]
                # 计算性能指标
                function.compute_segmentation_metric(segment_line_output, segment_line_labels, segment_line_loss_output, segmentation_metric_dict)
            function.save_segmentation_results(images.data.cpu().numpy(), segment_line_output.data.cpu().numpy())

        ## 检测
        if args.use_detection:
            # 计算输出以及loss
            detect_character_output = model_dict['detect_character'](features)
            if have_label:
                if args.gpu:
                    detect_character_labels = Variable(data[4].cuda(async=True))
                else:
                    detect_character_labels = Variable(data[4])
                detect_character_loss_output = loss_dict['detect_character'](detect_character_output, detect_character_labels)
                loss_gradient += detect_character_loss_output[0]
                # 计算性能指标
                function.compute_detection_metric(detect_character_output, detect_character_labels, detect_character_loss_output, detection_metric_dict)

            function.save_detection_results(data[0], images.data.cpu().numpy(), detect_character_output.data.cpu().numpy(), it=i)

            # break


    if have_label:
        print('Epoch: {:d} \t Phase: {:s}'.format(epoch, phase))
        function.print_metric('classification', classification_metric_dict)
        if args.use_segmention:
            function.print_segmentation_metric('segmentation', segmentation_metric_dict)
        if args.use_detection:
            function.print_detection_metric('detection', detection_metric_dict)



def train_eval(p_dict, phase='train'):
    ### 传入参数
    epoch = p_dict['epoch']
    model_dict = p_dict['model_dict']           # 模型
    loss_dict = p_dict['loss_dict']             # loss 函数
    if phase == 'train':
        data_loader = p_dict['train_loader']        # 训练数据
        optimizer = p_dict['optimizer']             # 优化器
    else:
        data_loader = p_dict['val_loader']

    ### 局部变量定义
    classification_metric_dict = dict()
    segmentation_metric_dict = dict()
    detection_metric_dict = dict()


    ### 模型训练
    for i,data in enumerate(tqdm(data_loader)):
        if i > 800:
            break
        if i == 0:
            function.save_middle_results(data)
        # images, labels = [Variable(x.cuda(async=True)) for x in data[1:3]]
        if args.gpu:
            images = Variable(data[1].cuda(async=True))
            labels = Variable(data[2].cuda(async=True))
        else:
            images = Variable(data[1])
            labels = Variable(data[2])

        # 提取特征
        features = model_dict['feature'](images)

        loss_gradient = 0

        ## 分类
        # 计算输出以及loss
        # outputs, all_outputs = model_dict['classify'](features)
        # classification_loss_output = loss_dict['classify'](outputs, labels, 1)                 # 分类loss
        # loss_gradient += classification_loss_output[0]
        # 计算性能指标
        # function.compute_metric(outputs, labels, classification_loss_output, classification_metric_dict)

        ## 分割
        if args.use_segmention:
            # 计算输出以及loss
            if args.gpu:
                segment_line_labels = Variable(data[3].cuda(async=True))
            else:
                segment_line_labels = Variable(data[3])
            segment_line_output = model_dict['segment_line'](features)
            segment_line_loss_output = loss_dict['segment_line'](segment_line_output, segment_line_labels)
            loss_gradient += segment_line_loss_output[0]
            # 计算性能指标
            function.compute_segmentation_metric(segment_line_output, segment_line_labels, segment_line_loss_output, segmentation_metric_dict)

        ## 检测
        if args.use_detection:
            # 计算输出以及loss
            if args.gpu:
                detect_character_labels = Variable(data[4].cuda(async=True))
            else:
                detect_character_labels = Variable(data[4])
            detect_character_output = model_dict['detect_character'](features)
            detect_character_loss_output = loss_dict['detect_character'](detect_character_output, detect_character_labels)
            loss_gradient += detect_character_loss_output[0]
            # 计算性能指标
            function.compute_detection_metric(detect_character_output, detect_character_labels, detect_character_loss_output, detection_metric_dict)


        # print(outputs.size(), labels.size(),data[3].size(),segment_line_output.size())
        # print('detection', detect_character_labels.size(), detect_character_output.size())
        # return

        # 训练阶段
        if phase == 'train':
            optimizer.zero_grad()
            loss_gradient.backward()
            optimizer.step()


    print('Epoch: {:d} \t Phase: {:s} \n'.format(epoch, phase))
    # function.print_metric('classification', classification_metric_dict)
    if args.use_segmention:
        function.save_segmentation_results(images.data.cpu().numpy(), segment_line_output.data.cpu().numpy())
        metric = function.print_segmentation_metric('segmentation', segmentation_metric_dict)
        print('segmentation metric:\t', metric, '\n')
    if args.use_detection:
        metric = function.print_detection_metric('detection', detection_metric_dict)
        if phase == 'val' and metric > p_dict['best_metric']:
            p_dict['best_metric'] = metric
            function.save_model(p_dict)
        print('best_metric:\t', p_dict['best_metric'], '\n')




def main():
    p_dict = dict() # 所有参数
    p_dict['args'] = args

    ### 加载数据
    word_index_dict = json.load(open(args.word_index_json))
    args.words = { v:k for k,v in word_index_dict.items() }
    p_dict['word_index_dict'] = word_index_dict
    num_classes = args.num_classes
    p_dict['num_classes'] = num_classes
    image_label_dict = json.load(open(args.image_label_json))
    p_dict['image_label_dict'] = image_label_dict
    # 划分数据集
    test_filelist = sorted(glob(os.path.join(args.data_dir,'test/plots','*.png')))
    trainval_filelist = sorted(glob(os.path.join(args.data_dir,'train/plots','*.png')))
    # 两种输入size训练
    # train_filelist1: 长宽比小于8:1的图片，经过padding后变成 64*512 的输入
    # train_filelist2: 长宽比大于8:1的图片，经过padding,crop后变成 64*1024的输入
    '''
    train_filelist1, train_filelist2 = [],[]
    # 黑名单，这些图片的label是有问题的
    black_list = set(json.load(open(args.black_json))['black_list'])
    image_hw_ratio_dict = json.load(open(args.image_hw_ratio_json))
    for f in trainval_filelist:
        image = f.split('/')[-1]
        if image in black_list:
            continue
        r = image_hw_ratio_dict[image]
        if r == 0:
            train_filelist1.append(f)
        else:
            train_filelist2.append(f)
    val_filelist = train_filelist1[-2048:]
    train_filelist = train_filelist1[:-2048] 
    '''
    # train_filelist1 = sorted(trainval_filelist)
    # val_filelist = train_filelist1[-2048:]
    # train_filelist = train_filelist1[:-2048] 
    val_filelist = test_filelist
    train_filelist = trainval_filelist
    # generated_list = glob(os.path.join(args.data_dir.replace('dataset', 'generated_images'), '*_image.png'))
    # n_test = 4096
    # pretrain_filelist = generated_list[:-n_test]
    # preval_filelist = generated_list[-n_test:]
    # train_filelist2 = train_filelist2
    image_size = args.input_shape
    test_dataset = dataloader.DataSet(
                test_filelist, 
                image_label_dict,
                num_classes, 
                # transform=train_transform, 
                args=args,
                image_size=image_size,
                phase='test')
    test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=args.workers, 
                pin_memory=True)
    train_dataset  = dataloader.DataSet(
                train_filelist, 
                image_label_dict, 
                num_classes, 
                image_size=image_size,
                args=args,
                phase='train')
    train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=args.workers, 
                pin_memory=True)
    val_dataset  = dataloader.DataSet(
                val_filelist, 
                image_label_dict, 
                num_classes, 
                image_size=image_size,
                args=args,
                phase='val')
    val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=args.batch_size,
                shuffle=False, 
                num_workers=args.workers, 
                pin_memory=True)
    '''
    pretrain_dataset = dataloader.DataSet(
                pretrain_filelist, 
                image_label_dict,
                num_classes, 
                image_size=image_size,
                word_index_dict = word_index_dict,
                args=args,
                font_range=[8,32],
                margin=10,
                rotate_range=[-10., 10. ],
                phase='pretrain')
    pretrain_loader = DataLoader(
                dataset=pretrain_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.workers, 
                pin_memory=True)
    preval_dataset = dataloader.DataSet(
                preval_filelist, 
                image_label_dict,
                num_classes, 
                image_size=image_size,
                word_index_dict = word_index_dict,
                args=args,
                font_range=[8,32],
                margin=10,
                rotate_range=[-10., 10. ],
                phase='pretrain')
    preval_loader = DataLoader(
                dataset=preval_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                num_workers=args.workers, 
                pin_memory=True)
    '''

    p_dict['train_loader'] = train_loader
    p_dict['val_loader'] = val_loader
    p_dict['test_loader'] = test_loader
    # p_dict['pretrain_loader'] = pretrain_loader


    # p_dict['train_loader'] = pretrain_loader
    # p_dict['val_loader'] = preval_loader
    # p_dict['test_loader'] = preval_loader




    ### 定义模型
    cudnn.benchmark = True
    framework.build_model(p_dict)
    parameters = []
    model_dict = p_dict['model_dict']
    for model in model_dict.values():
        for p in model.parameters():
            parameters.append(p)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    p_dict['optimizer'] = optimizer
    # model = torch.nn.DataParallel(model).cuda()
    # loss = Loss().cuda()
    start_epoch = 0
    # args.epoch = start_epoch
    # print ('best_f1score' + str(best_f1score))

    p_dict['epoch'] = 0
    p_dict['best_metric'] = 0


    ### 加载预训练模型与参数
    if os.path.exists(args.resume):
        function.load_model(p_dict, args.resume)


    ### 训练及测试模型
    if args.phase == 'test':
        # 测试输出文字检测结果
        test(p_dict)
    elif args.phase == 'train':

        best_f1score = 0
        eval_mode = 'eval'
        best_macc = 0
        p_dict['best_metric'] = 0
        for epoch in range(p_dict['epoch'] + 1, args.epochs):
            p_dict['epoch'] = epoch
            if best_f1score > 0.9:
                args.lr = 0.0001
            if best_f1score > 0.9:
                args.hard_mining = 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            train_eval(p_dict, 'train')
            train_eval(p_dict, 'val')


if __name__ == '__main__':
    main()
