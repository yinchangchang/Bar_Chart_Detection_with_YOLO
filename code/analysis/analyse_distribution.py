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
            print(k, vs)
            if 'ho' not in vs[1] :
                print(vs)
                cmd = 'cp {:s} ../../data'.format(fi_json.replace('.meta.json', '').replace('jsons', 'plots'))
                print(cmd)
                print(err)
            
            continue
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
    return bbox_list, vis


def analyse_distribution():
    file_list = glob('../../data/dataset/*/plots/*')
    print('There are {:d} images.'.format(len(file_list)))
    # key_set = set()
    x_delta_list = []
    y_delta_list = []
    for fi in tqdm(sorted(file_list)):
        fi_json = fi.replace('plots', 'jsons') + '.meta.json'
        bbox_list, x = parse_json(fi_json)
        # py_op.mywritejson(fi.replace('.png', '.bbox.json'), bbox_list)
        '''
        # draw a figure for a sample
        os.system('cp {:s} ../../data/'.format(fi))
        for bbox in bbox_list:
            xmin, ymin, xmax, ymax = bbox[:4]
            plt.plot([xmin, xmin], [ymin, ymax])
            plt.plot([xmax, xmax], [ymin, ymax])
            plt.plot([xmin, xmax], [ymin, ymin])
            plt.plot([xmin, xmax], [ymax, ymax])
        plt.show()
        return
        '''
        for bbox in bbox_list:
            xmin, ymin, xmax, ymax = bbox[:4]
            x_delta_list.append(xmax - xmin)
            y_delta_list.append(ymax - ymin)
    for delta_list in [x_delta_list, y_delta_list]:
        x_list = range(0, max(delta_list) + 1)
        x_dist = [0 for _ in x_list]
        for x in delta_list:
            x_dist[x] += 1
        plt.plot(x_list, x_dist)
        plt.show()


def main():
    analyse_distribution()

if __name__ == '__main__':
    main()
