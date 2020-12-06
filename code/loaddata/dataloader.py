# encoding: utf-8

"""
Read images and corresponding labels.
"""

import numpy as np
import os
import sys
import json
import torch
# import skimage
# from skimage import io
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from torch.utils.data import Dataset
import time

sys.path.append('loaddata')
import data_function, data_segmentation, data_detection

filters = [
            ImageFilter.SMOOTH,                 # 平滑，大于16可以用
            ImageFilter.SMOOTH_MORE,            # 平滑，大于16可以用
            ImageFilter.GaussianBlur(radius=1), # 大于16可以用

            ImageFilter.GaussianBlur(radius=2), # 大于32可以用
            ImageFilter.BLUR,                   # 大于32可以用
        ]

def histeq (im,nbr_bins =256):  
    # 对一副灰度图像进行直方图均衡化  
    #该函数有两个输入参数，一个是灰度图像，一个是直方图中使用小区间的数目  
    #函数返回直方图均衡化后的图像，以及用来做像素值映射的累计分布函数  
    # 计算图像的直方图  
    imhist,bins =np.histogram(im.flatten(),nbr_bins,normed=True)  
    cdf =imhist.cumsum() #cumulative distribution function  
    cdf =255*cdf/cdf[-1] #归一化，函数中使用累计分布函数的最后一个元素（下标为-1，目标是  
    # 将其归一化到0-1范围 ）  
    # 使用累计分布函数的线性插值，计算新的像素值  
    im2=np.interp(im.flatten(),bins[:-1],cdf) # im2 is an array  
    return im2.reshape(im.shape),cdf  

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
    size = d['figure_size'][0]
    for k,vs in d.items():
        if k == 'figure_size':
            # print(k, vs)
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
    new_bbox_list = []
    for b in bbox_list:
        xmin, ymin, xmax, ymax, l = b[:5]
        xmin = min(size[0], max(0, xmin))
        xmax = min(size[0], max(0, xmax))
        ymin = min(size[1], max(0, ymin))
        ymax = min(size[1], max(0, ymax))
        if xmax > xmin and ymax > ymin:
            nb = [ymin, xmin, ymax, xmax, l]
            new_bbox_list.append(nb)
    bbox_list = new_bbox_list
    return bbox_list



class DataSet(Dataset):
    def __init__(self, 
            image_names, 
            image_label_dict, 
            class_num, 
            transform=None, 
            image_size=None,        # 最后生成的图片大小
            word_index_dict=None,   # 字符与index的对应
            phase='train',          # phase
            args=None,              # 全局参数
            font_range=None,        # 生成字符大小范围
            rotate_range=None,      # 图片旋转范围
            margin=None             # 图片边缘不覆盖字符，以免旋转时候丢失
            ):

        self.font_range = font_range
        self.rotate_range = rotate_range
        self.margin = margin
        self.image_names = image_names
        self.image_label_dict = image_label_dict
        self.transform = transform
        self.phase = phase
        self.class_num = class_num
        self.word_labels = { }
        self.image_size = image_size
        self.word_index_dict = word_index_dict
        self.args = args
        self.stride = args.stride
        self.anchors = args.anchors
        if self.phase != 'pretrain':
            for image_name in image_names:
                image_name = image_name.split('/')[-1]
                if image_name not in image_label_dict:
                    try:
                        image_label_dict[image_name] = image_label_dict[image_name.replace('seg.','').split('.png')[0]+'.png']
                    except:
                        image_label_dict[image_name] = ''
                word_label = np.zeros(class_num)
                label = image_label_dict[image_name]
                for l in label.split():
                    word_label[int(l)] = 1
                self.word_labels[image_name] = word_label.astype(np.float32)

        self.data_dict = dict()
    def __getitem__(self, index):
        image_name = self.image_names[index]
        # print('load', image_name)
        no_aug = self.args.no_aug

        if 'dataset' in image_name:

            datadir = '/usr/local/data/det'
            ### 生成的图像
            image = Image.open(image_name).convert('RGB')
            seg = Image.open(image_name).convert('RGB')
            # h,w = image.size
            # h = int(h / self.args.stride)
            # w = int(w / self.args.stride)
            # seg = seg.resize((h, w))
            bbox = parse_json(image_name.replace('plots', 'jsons') + '.meta.json')

            while max(image.size) > 1.4 * max(self.args.input_shape) and self.args.phase != 'test':
                image = image.resize((int(image.size[0]/ 2), int(image.size[1]/2)))
                seg = seg.resize((int(seg.size[0]/ 2), int(seg.size[1]/2)))
                for i,b in enumerate(bbox):
                    bbox[i] = [int(b[0]/2), int(b[1]/2), int(b[2]/2), int(b[3]/2), b[4]]
            max_size = max(self.args.input_shape)
            if max(image.size) > max_size:
                nbl = []
                for i,b in enumerate(bbox):
                    b[2] = min(b[2], max_size)
                    b[3] = min(b[3], max_size)
                    if b[3] > b[1] and b[2] > b[0]:
                        nbl.append(b)
                bbox = nbl
            # label= json.load(open(image_name.replace('.png', '.label.json')))
            label = []


            # 数据转换
            image = data_function.image_to_numpy(image, self.args.input_shape).astype(np.float32)
            # print(image.shape)

            # 数据增广
            if self.phase == 'train':
                train_shape = [128, 128]
                # image, bbox = data_function.augment(image, bbox, train_shape)

            # 分类标签
            word_label = np.zeros(1)
            # word_label = np.zeros(self.class_num, dtype=np.float32)
            # for w in label:
            #     word_label[self.word_index_dict[w]] = 1

            # 分割标签
            # seg_label = data_segmentation.get_seg_label(seg, self.stride, bbox, self.args)
            seg_label = np.zeros_like(image)

            # 检测标签
            bbox_label, bbox_images = data_detection.get_bbox_label(image, bbox, self.stride, self.anchors, self.phase, self.args)
            bbox_images = np.array(bbox_images)


            image = (image / 128. - 1).astype(np.float32)

            # np.save(os.path.join(datadir, name + '.image.npy'), image)
            # np.save(os.path.join(datadir, name + '.bbox_label.npy'), bbox_label)
            # np.save(os.path.join(datadir, name + '.bbox_images.npy'), bbox_images)

            return image_name, torch.from_numpy(image), torch.from_numpy(word_label), seg_label, bbox_label, bbox_images

        elif 'generated_images' in image_name:
            ### 生成的图像
            image = Image.open(image_name)
            seg = Image.open(image_name.replace('_image.png', '_seg.png'))
            bbox = json.load(open(image_name.replace('_image.png', '_bbox.json')))
            label= json.load(open(image_name.replace('_image.png', '_label.json')))

            # 数据增广
            image, seg, bbox, label = data_function.augment(image, seg, bbox, label)

            # 数据转换
            image = data_function.image_to_numpy(image).astype(np.float32)

            # 分类标签
            word_label = np.zeros(self.class_num, dtype=np.float32)
            for w in label:
                word_label[self.word_index_dict[w]] = 1

            # 分割标签
            seg_label = data_segmentation.get_seg_label(seg, self.stride, bbox)

            # 检测标签
            bbox_label, bbox_images = data_detection.get_bbox_label(image, bbox, self.stride, self.anchors)
            bbox_images = np.array(bbox_images)


            image = (image / 128. - 1).astype(np.float32)
            return image_name, torch.from_numpy(image), torch.from_numpy(word_label), seg_label, bbox_label, bbox_images
        else:
            image, word_label = random_crop_image(image_name, self.image_label_dict[image_name.split('/')[-1]], self.image_size, self.class_num, self.phase, index, no_aug, self.args)
            return image_name, image, word_label


    def __len__(self):
        return len(self.image_names) 

last_random = 10
def get_random(idx):
    global last_random
    if last_random < 1:
        np.random.seed(int(last_random * 1000000 + time.time()) + idx)
    else:
        np.random.seed(int((time.time())))
    x = np.random.random()
    while np.abs(last_random - x) < 0.1:
        x = np.random.random()
    last_random = x
    return x

def random_crop_image(image_name, text, image_size, class_num, phase, idx, no_aug, args):
    # label
    text = text.split()
    word_label = np.zeros(class_num, dtype=np.float32)

    
    if args.hist:
        if get_random(idx+34) < 0.4 and phase == 'train':
            image = Image.open(image_name).convert('RGB')
        else:
            # 直方图均衡化
            image = Image.open(image_name).convert('YCbCr')
            image = np.array(image)
            imy = image[:,:,0]
            imy,_ = histeq(imy)
            image[:,:,0] = imy
            image = Image.fromarray(image, mode='YCbCr').convert('RGB')
    else:
        image = Image.open(image_name).convert('RGB')
    x = np.array(image)
    assert x.min() >= 0
    assert x.max() < 256

    if phase == 'train' and not no_aug:
        # 旋转
        if get_random(idx+11) < 0.8:
            theta = int(6 * get_random(idx+1)) - 3
            image = image.rotate(theta)

        # 模糊处理
        if get_random(idx+2) < 0.3:
            np.random.shuffle(filters)
            image = image.filter(filters[0])

        # 短边小于64， 直接填0
        h,w = image.size
        if w < image_size[1] and h > 64:
            if get_random(idx+3) < 0.3:
                image = np.array(image)
                start_index = int((image_size[1] - w)/2)
                new_image = np.zeros((image_size[1], h, 3), dtype=np.uint8)
                new_image[start_index:start_index+w, :, :] = image
                image = Image.fromarray(new_image)


    # 先处理成 X * 64 的图片
    h,w = image.size
    h = int(float(h) * image_size[1] / w)
    image = image.resize((h, image_size[1]))

    if phase == 'train' and not no_aug:

        # 放缩 0.8~1.2
        h,w = image.size
        r = get_random(idx+4) / 4. + 0.8
        image = image.resize((int(h*r), int(w*r)))

        # crop
        if min(h,w) > 32:
            crop_size = 20
            x = int((crop_size * get_random(idx+5) - crop_size/2) * r)
            y = int((crop_size * get_random(idx+6) - crop_size/2) * r)
            image = image.crop((max(0,x),max(0,y),min(0,x)+h,min(0,y)+w))

        # 有时需要生成一些低分辨率的图片
        h,w = image.size
        r = get_random(idx+7)
        
        # 从新变为 X * 64 的图片
        h = int(float(h) * image_size[1] / w)
        image = image.resize((h, image_size[1]))

    # 填充成固定大小
    image = np.transpose(np.array(image), [2,0,1]).astype(np.float32)
    if image.shape[2] < image_size[0]:
        # 长宽比例小于8(16)，直接填充
        if phase == 'test':
            # 正中间
            start = int(np.abs(image_size[0] - image.shape[2])/2)
        else:
            start = int(np.random.random() * np.abs(image_size[0] - image.shape[2]))
        new_image = np.zeros((3, image_size[1], image_size[0]), dtype=np.float32)
        new_image[:,:,start:start+image.shape[2]] = image
        if phase == 'test':
            new_image = np.array([new_image]).astype(np.float32)
        for w in text:
            word_label[int(w)] = 1
    else:
        # 长宽比例大于16，随机截取
        if phase == 'test':
            # 测试阶段直接合并
            crop_num = image.shape[2] * 2 / image_size[0] + 1
            new_image = np.zeros((crop_num, 3, image_size[1], image_size[0]), dtype=np.float32)
            for i in range(crop_num):
                start_index = i * image_size[0] / 2
                end_index = start_index + image_size[0]
                if end_index > image.shape[2]:
                    new_image[i,:,:,:image.shape[2] - start_index] = image[:,:,start_index:end_index]
                else:
                    new_image[i] = image[:,:,start_index:end_index]
            for w in text:
                word_label[int(w)] = 1
        else:
            # 训练阶段不算负例loss
            start = int(np.random.random() * np.abs(image_size[0] - image.shape[2]))
            new_image = image[:,:,start:start+image_size[0]]
            for w in text:
                word_label[int(w)] = -1

    image = new_image
    if phase == 'train':
        image = image.astype(np.float32)
        # 增加噪声
        if get_random(idx+8) < 0.1:
            noise_level = 64
            noise = np.random.random(image.shape) * noise_level - noise_level / 2.
            image = image + noise 
            # noise = np.random.random(image.shape[1:]) * noise_level - noise_level / 2.
            # image = image + np.array([noise, noise, noise])
            image = image.astype(np.float32)

    return image, word_label
