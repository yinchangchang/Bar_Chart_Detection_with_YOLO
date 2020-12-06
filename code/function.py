# coding=utf8
#########################################################################
# File Name: function.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月12日 星期三 14时28分43秒
#########################################################################

import os

import numpy as np

import torch

# file
import loaddata
import detection
from tools import parse
from loaddata import data_function
from detection import detect_function

args = parse.args

def save_model(p_dict, name='best.ckpt', folder='../data/models/'):
    if not os.path.exists(folder):
        os.mkdir(folder)
    model_dict = p_dict['model_dict']
    state_dict = dict()
    for k,v in model_dict.items():
        state_dict[k] = model_dict[k].state_dict()
        for key in state_dict[k].keys():
            state_dict[k][key] = state_dict[k][key].cpu()
    all_dict = {
            'epoch': p_dict['epoch'],
            'args': p_dict['args'],
            'best_metric': p_dict['best_metric'],
            'state_dict': state_dict 
            }
    torch.save(all_dict, os.path.join(folder, name))

def load_model(p_dict, model_file):
    all_dict = torch.load(model_file)
    p_dict['epoch'] = all_dict['epoch']
    p_dict['args'] = all_dict['args']
    p_dict['best_metric'] = all_dict['best_metric']
    for k,v in all_dict['state_dict'].items():
        p_dict['model_dict'][k].load_state_dict(all_dict['state_dict'][k])


def save_segmentation_results(images, segmentations, folder='../data/middle_segmentation'):
    stride = args.stride

    if not os.path.exists(folder):
        os.mkdir(folder)

    # images = images.data.cpu().numpy()
    # segmentations = segmentations.data.cpu().numpy()
    images = (images * 128) + 127
    segmentations[segmentations>0] = 255
    segmentations[segmentations<0] = 0

    # print(images.shape, segmentations.shape)
    for ii, image, seg in zip(range(len(images)), images, segmentations):
        image = data_function.numpy_to_image(image)
        new_seg = np.zeros([3, seg.shape[1] * stride, seg.shape[2] * stride])
        for i in range(seg.shape[1]):
            for j in range(seg.shape[2]):
                for k in range(3):
                    new_seg[k, i*stride:(i+1)*stride, j*stride:(j+1)*stride] = seg[0,i,j]
        seg = new_seg
        seg = data_function.numpy_to_image(seg)
        image.save(os.path.join(folder, str(ii) + '_image.png'))
        seg.save(os.path.join(folder, str(ii) + '_seg.png'))


def save_middle_results(data, folder = '../data/middle_images'):
    stride = args.stride

    if not os.path.exists(folder):
        os.mkdir(folder)
    numpy_data = [x.data.numpy() for x in data[1:]]
    data =  data[:1] + numpy_data
    image_names, images, word_labels, seg_labels, bbox_labels, bbox_images =  data[:6]
    images = (images * 128) + 127
    seg_labels = seg_labels*127 + 127


    for ii, name, image, seg, bbox_image in zip(range(len(image_names)), image_names, images, seg_labels, bbox_images):
        name = name.split('/')[-1]
        image = data_function.numpy_to_image(image)
        new_seg = np.zeros([3, seg.shape[1] * stride, seg.shape[2] * stride])
        # print(seg[0].max(),seg[0].min())
        for i in range(seg.shape[1]):
            for j in range(seg.shape[2]):
                for k in range(3):
                    new_seg[k, i*stride:(i+1)*stride, j*stride:(j+1)*stride] = seg[0,i,j]
        seg = new_seg
        seg = data_function.numpy_to_image(seg)
        # image.save(os.path.join(folder, name))
        # seg.save(os.path.join(folder, name.replace('image.png', 'seg.png')))
        image.save(os.path.join(folder, str(ii) + '_image.png'))
        seg.save(os.path.join(folder, str(ii) + '_seg.png'))

        for ib,bimg in enumerate(bbox_image):
            # print(bimg.max(), bimg.min(), bimg.dtype)
            bimg = data_function.numpy_to_image(bimg)
            bimg.save(os.path.join(folder, str(ii)+'_'+ str(ib) + '_bbox.png'))

def save_detection_results(names, images, detect_character_output, folder='../data/test_results/', it=0):
    stride = args.stride

    if not os.path.exists(folder):
        os.mkdir(folder)
    images = (images * 128) + 127

    for i, name, image, bboxes in zip(range(len(names)), names, images, detect_character_output):
        name = name.split('/')[-1]
        data_function.numpy_to_image(image).save(os.path.join(folder, name + '.image.png'))
        np.save(os.path.join(folder, name + '.bbox.npy'), bboxes)
        detected_bbox = detect_function.nms(bboxes)
        np.save(os.path.join(folder, name + '.bbox_after_nms.npy'), detected_bbox)
        image = data_function.add_bbox_to_image(image, detected_bbox)
        image.save(os.path.join(folder, name + '.bbox.png'))



def compute_detection_metric(outputs, labels, loss_outputs,metric_dict):
    loss_outputs[0] = loss_outputs[0].data
    metric_dict['metric'] = metric_dict.get('metric', []) + [loss_outputs]

def compute_segmentation_metric(outputs, labels, loss_outputs, metric_dict):
    loss_outputs[0] = loss_outputs[0].data
    metric_dict['metric'] = metric_dict.get('metric', []) + [loss_outputs]

def compute_metric(outputs, labels, loss_outputs,metric_dict):
        # loss_output_list, f1score_list, recall_list, precision_list):
    preds = outputs.data.cpu().numpy() > 0
    labels = labels.data.cpu().numpy()
    for pred, label in zip(preds, labels):
        pred[label<0] = -1
        if label.sum() < 0.5:
            continue
        tp = (pred + label == 2).sum()
        tn = (pred + label == 0).sum()
        fp = (pred - label == 1).sum()
        fn = (pred - label ==-1).sum()
        precision = 1.0 * tp / (tp + fp + 10e-20)
        recall   = 1.0 * tp / (tp + fn + 10e-20)
        f1score = 2. * precision * recall / (precision + recall + 10e-20)

        metric_dict['precision'] = metric_dict.get('precision', []) + [precision]
        metric_dict['f1score'] = metric_dict.get('f1score', []) + [f1score]
        metric_dict['recall'] = metric_dict.get('recall', []) + [recall]
    metric_dict['loss'] = metric_dict.get('loss', []) +  [[x.data.cpu().numpy() for x in loss_outputs]]

def print_metric(first_line, metric_dict):
    print(first_line)
    loss_array = np.array(metric_dict['loss']).mean(0)
    f1score_list = metric_dict['f1score']
    recall_list = metric_dict['recall']
    precision_list = metric_dict['precision']
    if len(loss_array) >= 3:
        print('loss: {:3.4f}\t pos loss: {:3.4f}\t negloss: {:3.4f}'.format(loss_array[0], loss_array[1], loss_array[2]))
    else:
        print('loss: {:3.4f}'.format(loss_array[0]))
    if f1score_list is not None:
        print('f1score: {:3.4f}\t recall: {:3.4f}\t precision: {:3.4f}'.format(np.mean(f1score_list), np.mean(recall_list), np.mean(precision_list)))
    print('\n')

def print_detection_metric(first_line, metric_dict):
    metrics = np.asarray(metric_dict['metric'], np.float32)
    print(first_line)
    print(
        'tpr %3.4f, tnr %3.4f, total pos %d, total neg %d,      multi-classification accuracy %3.4f'
        % (np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
           np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
           np.sum(metrics[:, 7]), np.sum(metrics[:, 9]),
           np.sum(metrics[:, 11]) / np.sum(metrics[:, 12]),
           ))
    print(
        'loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f,  multi-classification loss, %2.4f\n'
        % (np.mean(metrics[:, 0]), np.mean(metrics[:, 1]),
           np.mean(metrics[:, 2]), np.mean(metrics[:, 3]),
           np.mean(metrics[:, 4]), np.mean(metrics[:, 5]),
           np.mean(metrics[:, 10])
           ))
    dice = np.sum(metrics[:, 6]) / ( np.sum(metrics[:, 7]) +  np.sum(metrics[:, 9]) - np.sum(metrics[:, 8]))
    return np.sum(metrics[:, 11]) / np.sum(metrics[:, 12]) * dice

def print_segmentation_metric(first_line, metric_dict):
    metrics = np.asarray(metric_dict['metric'], np.float32)
    print(first_line)
    print(
        'tpr %3.4f, tnr %3.4f, total pos %d, total neg %d,      multi-classification accuracy %3.4f'
        % (np.sum(metrics[:, 2]) / np.sum(metrics[:, 3]),
           np.sum(metrics[:, 4]) / np.sum(metrics[:, 5]),
           np.sum(metrics[:, 3]), np.sum(metrics[:, 5]),
           np.sum(metrics[:, 7]) / np.sum(metrics[:, 8]),
           ))
    print(
        'loss %2.4f, binary loss %2.4f, multi-classification loss, %2.4f\n'
        % (np.mean(metrics[:, 0]), np.mean(metrics[:, 1]),
           np.mean(metrics[:, 6])
           ))
    dice = np.sum(metrics[:, 2]) / ( np.sum(metrics[:, 3]) +  np.sum(metrics[:, 5]) - np.sum(metrics[:, 4]))
    return np.sum(metrics[:, 7]) / np.sum(metrics[:, 8]) * dice
