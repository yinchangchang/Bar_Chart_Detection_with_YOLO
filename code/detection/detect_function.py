# coding=utf8
#########################################################################
# File Name: detect_function.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 14时35分03秒
#########################################################################

import sys
import numpy as np
sys.path.append('../')

from loaddata import data_function, data_detection
from tools import parse

args = parse.args


def nms(bboxes):
    # 生成中心结果
    stride = args.stride
    anchors = args.anchors
    neg_th = args.neg_th
    nms_ol = args.nms_ol
    
    # xx = np.arange(bboxes.shape[0]) * stride + stride/2
    # yy = np.arange(bboxes.shape[1]) * stride + stride/2
    # points = np.array(np.meshgrid(yy, xx)).transpose(1,2,0)

    # 找到正例bbox
    is_bbox = bboxes[:,:,:,0]
    is_bbox = 1 - 1.0 / (1 + np.exp(is_bbox))
    indices = np.where(is_bbox > neg_th)
    predicted_bbox = []
    for ix,iy,ia in zip(indices[0], indices[1], indices[2]):
        probability = bboxes[ix,iy,ia,0]
        dx,dy,dw,dh = bboxes[ix,iy,ia,1:5]
        logit = bboxes[ix,iy,ia,5:]
        # print('bboxes', bboxes.shape)
        # print('points', points.shape)
        # print(ix, iy)
        # mx,my = points[ix,iy]
        mx,my = (ix + 0.5) * stride, (iy + 0.5) * stride # 中心点
        w,h = anchors[ia]

        predict = np.argmax(logit)
        px,py = mx+dx*w,my+dy*h     # 中心点
        pw = np.exp(dw) * w         # 宽
        ph = np.exp(dh) * h         # 高
        bbox = [probability, px - pw/2, py - ph/2, px + pw/2, py + ph/2, predict]
        predicted_bbox.append(bbox)

    predicted_bbox = sorted(predicted_bbox, reverse=True)
    selected_bbox = []
    for bbox in predicted_bbox:
        if len(selected_bbox) == 0:
            selected_bbox.append(bbox)
        else:
            vis = 0
            for sb in selected_bbox:
                if sb[-1] == bbox[-1]:
                    iou = data_detection.comput_iou(sb[1:5], bbox[1:5])
                    if iou > 0.01:
                        vis = 1
                        break
                else:
                    iou = data_detection.comput_iou(sb[1:5], bbox[1:5])
                    if iou > 0.01:
                        vis = 1
                        break
            if vis == 0:
                selected_bbox.append(bbox)
    return selected_bbox



def add_bbox_to_image(image, bbox):
    pass
