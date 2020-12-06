# coding=utf8
#########################################################################
# File Name: data_segmentation.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月12日 星期三 13时35分21秒
#########################################################################

import os
import sys
import time
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter

sys.path.append('loaddata')
import data_function

def get_seg_label(seg, stride, bbox, args):
    '''
    二分类分割标签
    '''
    seg = data_function.image_to_numpy(seg, args.input_shape).astype(np.float32)
    start = int(stride / 2)
    seg = seg[:2, start:, start:]       # 二分类，只取第一维数据即可

    # 第一维度是否有物体，二分类
    seg = seg[:, ::stride, ::stride]
    seg[seg<192] = 1
    seg[seg>192] = 0
    seg[1] = -1

    # 第二维度多分类
    seg[0][seg[0]==1] = -1
    seg[1,:,:] = -1
    for bb in bbox:
        sx,sy,ex,ey = [int(x/stride) for x in bb[:4]]
        c = bb[-1]
        # if ex-sx > 3 and ey-sy > 3:
        #     seg[0, sx+2:ex-1, sy+2:ey-1] = 1
        #     seg[1, sx+2:ex-1, sy+2:ey-1] = c
        if ex-sx > 2 and ey-sy > 2:
            seg[0, sx+1:ex-1, sy+1:ey-1] = 1
            seg[1, sx+1:ex-1, sy+1:ey-1] = c

    return seg

