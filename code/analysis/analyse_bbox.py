# coding=utf8
import time
import sys
import os

import numpy as np
import json
from tqdm import tqdm


import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
sys.path.append('../tools')
import parse, py_op
args = parse.args

def parse_json(fi_json):
    d = json.load(open(fi_json))
    # key_set = key_set | set(d.keys())
    bbox_list = []
    key_idx_dict = {
            'figure_size': 0,
            'axis_label_bbox': 1,
            'bar_bbox': 2,
            'tick_bbox': 3,
            'title_bbox': 4,
            }
    vis = 0
    for k,vs in d.items():
        if k == 'figure_size':
            # print(k, vs)
            size = vs[0]
            if 'ho' not in vs[1] :
                # print(vs)
                cmd = 'cp {:s} ../../data'.format(fi_json.replace('.meta.json', '').replace('jsons', 'plots'))
                # print(cmd)
                # print(err)
            
        elif k == 'tick_bbox':
            if len(vs):
                assert type(vs) is list
                for v in vs:
                    assert type(v) is list
                    for vi in v:
                        assert type(vi) is dict
                        assert len(vi) == 2
                        bbox = vi['bbox']
                        text = vi['text']
                        # bbox_list.append([key_idx_dict[], bbox, text])
                        bbox_list.append(bbox + [key_idx_dict[k], k, text])
                # vis = 1
        elif k == 'axis_label_bbox':
            if len(vs):
                # print(k, vs)
                assert type(vs) is list
                for v in vs:
                    assert type(v) is dict
                    assert len(v) == 2
                    bbox = v['bbox']
                    text = v['text']
                    # bbox_list.append([k, bbox, text])
                    bbox_list.append(bbox + [key_idx_dict[k], k, text])
                # vis = 1
        elif k == 'bar_bbox':
            if len(vs):
                for v in vs:
                    assert type(v) == dict
                    assert len(v) == 2
                    bbox = v['bbox']
                    height = v['height']
                    # bbox_list.append([k, bbox, height])
                    bbox_list.append(bbox + [key_idx_dict[k], k, height])
        elif k == 'title_bbox':
            if len(vs):
                assert type(vs) is dict
                assert len(vs) == 2
                bbox = vs['bbox']
                text = vs['text']
                # bbox_list.append([k, bbox, text])
                bbox_list.append(bbox + [key_idx_dict[k], k, text])
        else:
            assert KeyError('New key: ' + k)
    bbox_list = [b[:5] for b in bbox_list]
    return bbox_list, size


def analyse_bbox():
    file_list = glob('../../data/dataset/*/plots/*')
    print('There are {:d} images.'.format(len(file_list)))
    # key_set = set()
    x_delta_list = [1, 2, 4]
    y_delta_list = [8, 16, 32]
    anchors = []
    for x in x_delta_list:
        anchors.append([x,x])
        for y in y_delta_list:
            anchors.append([x,y])
            anchors.append([y,x])
    n_bbox = [0 for _ in range(5)]
    n_valid = [0 for _ in range(5)]
    for fi in tqdm(sorted(file_list)):
        fi_json = fi.replace('plots', 'jsons') + '.meta.json'
        bbox_list, size = parse_json(fi_json)
        for b in bbox_list:
            assert len(b) == 5
        while max(size) >= 350:
            size = [size[0] / 2, size[1] / 2]
            nbl = []
            for b in bbox_list:
                b = [int(v / 2) for v in b[:4]] + [b[4]]
                nbl.append(b)
            bbox_list = nbl

        delta = 8
        for b in bbox_list:
            xmin, ymin, xmax, ymax, l = b
            so = (ymax - ymin) * (xmax - xmin)
            if so <= 0:
                continue
            n_bbox[l] += 1
            vis = 0
            for x in range(int(delta / 2), int(size[0]), delta):
                if vis: break
                for y in range(int(delta / 2), int(size[1]), delta):
                    if vis: break
                    for dx, dy in anchors:
                        if vis: break
                        sa = 8 * 8 * dx * dy

                        xmin_a = x - delta / 2 * dx
                        xmax_a = x + delta / 2 * dx
                        ymin_a = y - delta / 2 * dy
                        ymax_a = y + delta / 2 * dy

                        xmin_j = max(xmin, xmin_a)
                        xmax_j = min(xmax, xmax_a)
                        ymin_j = max(ymin, ymin_a)
                        ymax_j = min(ymax, ymax_a)
                        if ymax_j > ymin_j and xmax_j > xmin_j:
                            sj = (ymax_j - ymin_j) * (xmax_j - xmin_j)
                            if sj > 0.2 * (sa + so - sj):
                                vis = 1
            if vis: n_valid[l] += 1
            # print(float(n_valid) / n_bbox)
            print(['{:0.3f}'.format(float(nv+0.001) / (nb + 0.001)) for nv,nb in zip(n_valid, n_bbox)])






def main():
    analyse_bbox()

if __name__ == '__main__':
    main()
