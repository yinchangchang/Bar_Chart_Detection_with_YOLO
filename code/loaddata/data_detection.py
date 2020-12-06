# coding=utf8
#########################################################################
# File Name: data_detection.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月12日 星期三 13时35分29秒
#########################################################################
import sys
sys.path.append('loaddata')

import numpy as np

import data_function



def comput_iou(bbox, proposal):
    gsx,gsy,gex,gey = bbox[:4]
    psx,psy,pex,pey = proposal


    overlap_x =  max(min(pex, gex) - max(psx, gsx), 0)
    overlap_y =  max(min(pey, gey) - max(psy, gsy), 0)
    # 面积

    sp = (pex - psx) * (pey - psy)
    sg = (gex - gsx) * (gey - gsy)
    so = overlap_x * overlap_y
    iou = float(so) / (sg + sp - so)

    if iou > 0.4:
        delta_x = gex - gsx
        delta_y = gey - gsy
        if delta_x / delta_y > 3:
            ux = max(gex, pex) - min(gsx, psx)
            ix = min(gex, pex) - max(gsx, psx)
            if float(ix) / ux < 0.65:
                iou -= 0.2
        if delta_y / delta_x > 3:
            uy = max(gey, pey) - min(gsy, psy)
            iy = min(gey, pey) - max(gsy, psy)
            if float(iy) / uy < 0.65:
                iou -= 0.2


    return iou

# def get_bbox_label(image, font_place, font_size, font_num, args, image_size):
def get_bbox_label(image, bbox, stride, anchors, phase, args):
    assert len(bbox) > 0
    _,imgw,imgh = image.shape

    bbox_images = [image.copy() for _ in anchors]

    bbox_label = np.zeros((
        int(imgw/stride),
        int(imgh/stride), 
        len(anchors),          # 4
        6                           # 01, dx,dy,dw,dh, class
        ), dtype=np.float32)
    bbox_label[:, :, :, 0] = -1

    # print(bbox_label.shape, image.shape)

    # print bbox_label.shape
    words = []
    valid_ids = set()

    xya = args.xya
    ids = list(range(len(xya)))
    np.random.shuffle(ids)
    if phase == 'train':
        ids = ids[: int(len(xya)/256)]
        xya = xya[ids]
    elif 'val' in phase:
        ids = ids[: int(len(xya)/128)]
        xya = xya[ids]
    else:
        ids = ids[: int(len(xya)/128)]
        xya = xya[ids]
    # for ix in x_range:
    #     for iy in y_range:
    #         for ia in a_range:
    for ix,iy,ia in xya:
                proposal = [
                        ix*stride + int(stride / 2) - int(anchors[ia][0] / 2), 
                        iy*stride + int(stride / 2) - int(anchors[ia][1] / 2), 
                        ix*stride + int(stride / 2) + int(anchors[ia][0] / 2), 
                        iy*stride + int(stride / 2) + int(anchors[ia][1] / 2), 
                        ]    # (sx, sy, ex, ey)
                if ix >= bbox_label.shape[0]:
                    continue
                if iy >= bbox_label.shape[1]:
                    continue
                iou_fi = []
                for fi, font in enumerate(bbox):
                    iou = comput_iou(font, proposal)
                    iou_fi.append((iou, fi))
                max_iou, max_fi = sorted(iou_fi)[-1]
                # print(max_iou, max_fi)
                # print(iou_fi, bbox)
                if max_iou > args.pos_iou:
                    # 正例
                    font = bbox[max_fi]
                    if font[4] == 2:
                        valid_ids.add(max_fi)
                    dx = (font[0] + font[2] - proposal[0] - proposal[2]) / 2.0 / float(anchors[ia][0])      # 中心点的偏离比例
                    dy = (font[1] + font[3] - proposal[1] - proposal[3]) / 2.0 / float(anchors[ia][1])      # 中心点的偏离比例
                    gw = font[2] - font[0]
                    gh = font[3] - font[1]
                    dw = np.log(gw / float(anchors[ia][0])) 
                    dh = np.log(gh / float(anchors[ia][1]))
                    bbox_label[ix,iy,ia] = [1, dx, dy, dw, dh, font[4]]
                    words.append(font[4])
                    bbox_images[ia] = data_function.add_line(bbox_images[ia], bbox[max_fi])
                    bbox_images[ia] = data_function.add_line(bbox_images[ia], proposal, 0, 1)
                elif max_iou > args.neg_iou:
                    # 忽略
                    bbox_label[ix,iy,ia,0] = -1
                else:
                    # 负例
                    bbox_label[ix,iy,ia,0] = 0
            # print()
    # print(words) 
    n_bar = len([1 for b in bbox if b[4] == 2])
    # print(len(valid_ids), n_bar)
    for ib,b in enumerate(bbox):
        # if b[4] == 2 and ib not in valid_ids:
        if 1:
            xmin, ymin, xmax, ymax = b[:4]
            # print(sorted([xmax - xmin, ymax - ymin]))

    # 随机去掉一部分
    '''
    pos_index = np.where(bbox_label[:,:,:,0] == 1)
    select_index = data_function.random_select_indices(pos_index, len(pos_index[0]) - 32)
    bbox_label[select_index] = -1
    neg_index = np.where(bbox_label[:,:,:,0] == 0)
    select_index = data_function.random_select_indices(neg_index, len(neg_index[0]) - 128)
    bbox_label[select_index] = -1

    # pos_index = np.where(bbox_label[:,:,:,0] == 1)
    # neg_index = np.where(bbox_label[:,:,:,0] == 0)
    # print(len(pos_index[0]))
    # print(len(neg_index[0]))
    # print(err)
    '''


    return bbox_label, bbox_images

