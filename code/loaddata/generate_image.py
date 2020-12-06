# coding=utf8
#########################################################################
# File Name: generate_image.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月11日 星期二 19时53分04秒
#########################################################################


import numpy as np
import os
import json
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import time
import sys
sys.path.append('../tools')
import parse,py_op
args = parse.args

# files
import data_function



filters = [
            ImageFilter.SMOOTH,                 # 平滑，大于16可以用
            ImageFilter.SMOOTH_MORE,            # 平滑，大于16可以用
            ImageFilter.GaussianBlur(radius=1), # 大于16可以用

            ImageFilter.GaussianBlur(radius=2), # 大于32可以用
            ImageFilter.BLUR,                   # 大于32可以用
        ]
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

def comput_iou(font, proposal):
    fx,fy,fh,fw = font
    px,py,pd = proposal
    overlap_x =  max(min(pd, fh) - np.abs(fx - px), 0)
    overlap_y =  max(min(pd, fw) - np.abs(fy - py), 0)
    # 面积
    sf = fh * fw
    sp = pd * pd
    so = overlap_x * overlap_y
    iou = float(so) / (sf + sp - so)
    return iou

def get_resize_para(size, idx):
    if size > 48:
        rh, rw = 4,4
    elif size > 32:
        if idx % 2:
            rh, rw = 2,4
        else:
            rh, rw = 4,2
    elif size > 16:
        if idx % 2:
            rh, rw = 1,2
        else:
            rh, rw = 2,1
    else:
        return 1,1

    rhs = list(range(rh))
    np.random.seed(int(time.time()) + idx + 1)
    np.random.shuffle(rhs)
    rh = rhs[0] + 1

    rws = range(rw)
    np.random.seed(int(time.time()) + idx + 2)
    np.random.shuffle(rws)
    rw = rws[0] + 1

    return rh, rw

# def generate_image(idx, image, word_index_dict, class_num, args, image_size, no_aug):
def generate_image( idx, image, no_aug, dataset, my_text=None):
    '''
    这里的注释，默认参数是
        image_size [512, 64]
        rotate_range [-5, 5]
        font_range [8,32]
    '''

    word_index_dict = dataset['word_index_dict']
    class_num = dataset['class_num']
    image_size = dataset['image_size']
    font_range = dataset['font_range']
    rotate_range = dataset['rotate_range']
    margin = dataset['margin']


    ### 选择文字背景
    image = image.resize((1024,1024))
    h,w = image.size
    # 随机crop一个部分，resize成固定大小，会对文字有一定的水平竖直方向拉伸
    h_crop = int(get_random(idx + 10) * image_size[0] * 2 / 8) + image_size[0] * 6 / 8 # 长度范围 [374, 512]
    w_crop = int(get_random(idx + 11) * image_size[1] * 2 / 8) + image_size[1] * 6 / 8 # 宽度范围 [48, 64]
    if no_aug:
        h_crop = image_size[0] # 512
        w_crop = image_size[1] # 64
    # 选择文字背景，随机选择crop起始位置
    x = int(get_random(idx+12) * (h - h_crop))
    y = int(get_random(idx+13) * (w - w_crop))
    image = image.crop((x,y,x+h_crop,y+w_crop))

    # print(image.size)
    # return


    # 字体大小是最容易引起错误的变量，字体大小不能超出图片中心区域大小
    size = font_range[0] + int(get_random(idx+20) * (font_range[1] - font_range[0]))
    size = min(size, h_crop - 2*margin - 2, w_crop - 2*margin - 2)
    # print('size',size)

    # 字体数量，超过可容纳数量的一半以上，至少包含一个字符
    large_num = max(0, (h_crop - 2 * margin)/ size - 1)     
    word_num = int(min(large_num / 2, 5) + get_random(idx+21) * large_num / 2) + 1
    # word_num = int(large_num / 2 + get_random(idx+21) * large_num / 2) + 1
    word_num = max(0, word_num)
    # word_num = min(2, word_num)

    # 添加字体位置，并生成label信息
    # print('crop',h_crop,w_crop)
    place_x = int(get_random(idx+22) * (h_crop - word_num * size - margin)) 
    # print('x', place_x)
    if margin == 0:
        # 用于添加两排文字
        place_y = int(get_random(idx+23) * (w_crop/2 - size - margin)) + margin
    else:
        place_y = int(get_random(idx+23) * (w_crop - size - margin)) + margin
    place = (place_x, place_y)
    # print('place',place)
    label = np.zeros(class_num).astype(np.float32)

    # 随机增加文本
    text = u''
    indices = []
    words = list(word_index_dict.keys())
    if margin == 0:
        # 两排文字
        word_num *= 2
    while len(text) < word_num:
        np.random.shuffle(words)
        w = words[len(text)]
        if w in u'"(),.\' 0123456789<>=+-_!@#$%^&*()[]{}~':
            # 部分字符不建议生成
            continue
        text = text + w
        index = word_index_dict[w]
        label[index] = 1
        indices.append(index)

    # 得到bbox_label
    # bbox_label, seg_label = generate_bbox_label(image, place, size, word_num, args, image_size)
    bbox, seg , bbox_image= data_function.generate_bbox_seg(image, place, size, indices)


    # 字体，可以添加其他字体
    fonts = ['../../files/ttf/simsun.ttf']
    np.random.shuffle(fonts)
    font = fonts[0]

    # 颜色
    r = get_random(idx+24)
    if no_aug or r < 0.7:
        # 选择不同程度的黑色
        if r < 0.3:
            c = int(get_random(idx + 25) * 64)
            color = (c,c,c)
        else:
            rgb = 64
            r = int(get_random(idx + 27) * rgb)
            g = int(get_random(idx + 28) * rgb)
            b = int(get_random(idx + 29) * rgb)
            color = (r,g,b)
    else:
        # 随机颜色，但是选择较暗的颜色
        rgb = 256
        r = int(get_random(idx + 27) * rgb)
        g = int(get_random(idx + 28) * rgb)
        b = int(get_random(idx + 29) * rgb)
        ra = get_random(idx + 30)
        if ra < 0.5:
            ra = int(1000 * ra) % 3
            if ra == 0:
                r = 0
            elif ra == 1:
                g = 0
            else:
                b = 0
        color = (r,g,b)

    # 亮背景
    light_bg = 1
    if light_bg:
        image = np.array(image)
        # image = 128 + image / 2
        image[image<192] = 192
        image = image.astype(np.uint8)
        image = Image.fromarray(image)


    # 增加文字到图片
    if margin == 0:
        image = add_text_to_img(image, text[:word_num/2], size, font, color, place)
        image = add_text_to_img(image, text[word_num/2:], size, font, color, (place[0], place[1]+image_size[1]/2))
    else:
        image = add_text_to_img(image, text, size, font, color, place)
        seg = add_text_to_img(seg, text, size, font, (0,0,0), place)
        bbox_image= add_text_to_img(bbox_image, text, size, font, (0,0,0), place)



    # 还原成 [512, 64] 的大小
    # image = image.resize(image_size)


    # 最后生成图片后再一次旋转，图片模糊化
    if 0 and (get_random(idx+50) < 0.8 and not no_aug):

        # 旋转
        rotate_size = rotate_range[0] + int(get_random(idx+32) * (rotate_range[1] - rotate_range[0]))
        theta = int(rotate_size * 2 * get_random(idx+33)) - rotate_size
        image = image.rotate(theta)
        # 作分割的时候，标签信息也需要一起旋转
        seg_label = np.array([seg_label, seg_label, seg_label]) * 255
        seg_label = np.array(Image.fromarray(seg_label.transpose([1,2,0]).astype(np.uint8)).rotate(theta))
        seg_label = (seg_label[:,:,0] > 128).astype(np.float32)

    filters = [
            ImageFilter.SMOOTH,                 # 平滑，大于16可以用
            ImageFilter.SMOOTH_MORE,            # 平滑，大于16可以用
            ImageFilter.GaussianBlur(radius=1), # 大于16可以用

            ImageFilter.GaussianBlur(radius=2), # 大于32可以用
            ImageFilter.BLUR,                   # 大于32可以用
            ImageFilter.GaussianBlur(radius=2), # 多来两次
            ImageFilter.BLUR,                   # 多来两次
            ]

    # 当文字比较大的时候，增加一些模糊
    if 0 and size > 16:
        if size < 32:
            filters = filters[:3]
        np.random.shuffle(filters)
        image = image.filter(filters[idx % len(filters)])

    # add noise
    add_noise = 0
    if 0 and add_noise:
        noise_level = 32
        image = np.array(image)
        noise = np.random.random(image.shape) * noise_level - noise_level / 2.
        image = image + noise
        image = image.astype(np.uint8)
        image = Image.fromarray(image)


    # 有时候需要低分辨率的图片
    # resize_0, resize_1 = get_resize_para(size, idx)
    # image = image.resize([int(image_size[0]/resize_0), int(image_size[1]/resize_1)])

    # 还原成 [512, 64] 的大小
    image = image.resize(image_size)

    # return image, label, bbox_label, seg_label, size
    return image, list(text), bbox, seg, bbox_image, size

def add_text_to_img(img, text, size, font, color, place):
    imgdraw = ImageDraw.Draw(img)
    imgfont = ImageFont.truetype(font,size=size)
    imgdraw.text(place, text, fill=color, font=imgfont)
    return img


def generate_image_list(folder, number):
    if not os.path.exists(folder):
        os.mkdir(folder)
    image = Image.open('../../files/0000045_001.png').convert('RGB')
    dataset = { }
    dataset['word_index_dict'] = json.load(open('../../files/character_index_dict.json'))
    dataset['class_num'] = max(dataset['word_index_dict'].values()) + 1
    dataset['image_size'] = [256, 64]
    dataset['font_range'] = [26, 32]
    dataset['rotate_range'] = [-5, 5]
    dataset['margin'] = 10
    for i in range(number):
        print(i)
        image, text, bbox_label, seg_label, bbox_image, font_size = generate_image(i + int(time.time()), image, 1, dataset)
        image.save(os.path.join(folder, '{:d}_image.png'.format(i)))
        seg_label.save(os.path.join(folder, '{:d}_seg.png'.format(i)))
        bbox_image.save(os.path.join(folder, '{:d}_bbox.png'.format(i)))
        py_op.mywritejson(os.path.join(folder, '{:d}_bbox.json'.format(i)), bbox_label)
        py_op.mywritejson(os.path.join(folder, '{:d}_label.json'.format(i)), text)

        y,x = image.size # 512, 64
        for bbox in bbox_label:

            sx,sy,ex,ey,_ = bbox

            # print('bbox size', bbox, image.size, i)
            assert sx <= x
            assert ex <= x
            assert sy <= y
            assert ey <= y



def main():
    generate_image_list('../../data/generated_images/', 40000)

if __name__ == '__main__':
    main()
