# coding=utf8
#########################################################################
# File Name: segmentation/segment_function.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月13日 星期四 10时58分03秒
#########################################################################

import torch

def select(neg_output, neg_labels, indices):
    n = int(indices.sum().data.cpu().numpy())
    _, idcs = torch.topk(indices, n)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    # print('multi', neg_output.size(), neg_labels.size())
    return neg_output, neg_labels
