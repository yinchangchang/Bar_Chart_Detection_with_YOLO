# coding=utf8
#########################################################################
# File Name: detect_loss.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月13日 星期四 13时30分59秒
#########################################################################


import sys
import numpy as np
sys.path.append('detection/')

# torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# files
import detect_function



def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    # print(idcs)
    # print(err)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

def random_select(data, n):
    n_data = len(data[0])
    idx_list = list(range(n_data))
    np.random.shuffle(idx_list)
    idx_list = idx_list[:n]
    idcs = torch.from_numpy(np.array(idx_list)).cuda()
    new_data = []
    for d in data:
        new_d = torch.index_select(d, 0, idcs)
        new_data.append(new_d)
    return new_data

def select(neg_output, neg_labels, indices):
    n = int(indices.sum().data.cpu().numpy())
    _, idcs = torch.topk(indices, n)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    # print('multi', neg_output.size(), neg_labels.size())
    return neg_output, neg_labels

class Loss(nn.Module):
    def __init__(self, num_hard=32):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.m_loss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        print('------------------------------------------------------------------------------------------------')
        print('Use hard mining = ', self.num_hard)
        print('------------------------------------------------------------------------------------------------')

    def forward(self, output, labels, train=True):
        '''
        二分类检测
        output: [bs, w, h, n_anchors, 5+n_class]
        labels: [bs, w, h, n_anchors, 5+1]
        '''
        batch_size = labels.size(0)
        output = output.view(-1, output.size(4))
        output_m = output[:, 5:]
        output = output[:, :5]
        labels = labels.view(-1, 6)
        labels_m = labels[:, 5:]
        labels = labels[:, :5]

        # print(labels.shape)
        # print(output.shape)
        # print(labels_m.shape)
        # print(output_m.shape)

        pos_idcs = labels[:, 0] == 1
        # print('pos_idcs', pos_idcs.size(), pos_idcs.data.cpu().numpy().sum())
        pos_idcs_m = pos_idcs
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)
        # pos_output, pos_labels = random_select([pos_output, pos_labels], 32 * batch_size)

        # print('pos', pos_output.size(), pos_labels.size())

        neg_idcs = labels[:, 0] == 0
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]
        # neg_output, neg_labels = random_select([neg_output, neg_labels], 128 * batch_size)

        # print('neg', neg_output.size(), neg_labels.size())


        # 多分类
        '''
        labels_m = labels_m[pos_idcs_m].view(-1).long()
        pos_idcs_o = pos_idcs_m.unsqueeze(1).expand(pos_idcs_m.size(0), output_m.size(1))
        output_m = output_m[pos_idcs_o].view(labels_m.size(0), -1)
        '''

        output_m, labels_m = select(output_m, labels_m, pos_idcs_m.float())
        labels_m = labels_m.view(-1).long()

        # output_m, labels_m = random_select([output_m, labels_m], 8 * batch_size)
        # print('labels_m', labels_m.size())
        # print('output_m', output_m.size())

        if self.num_hard > 0:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        # neg_output, neg_labels = hard_mining(neg_output, neg_labels, 5 * batch_size)
        neg_prob = self.sigmoid(neg_output)

        #classify_loss = self.classify_loss(
        #   torch.cat((pos_prob, neg_prob), 0),
        #  torch.cat((pos_labels[:, 0], neg_labels + 1), 0))
        if len(pos_output) > 0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz), self.regress_loss(ph, lh),
                self.regress_loss(pw, lw), self.regress_loss(pd, ld)
            ]
            regress_losses_data = [l.data for l in regress_losses]
            classify_loss = 0.5 * self.classify_loss(
                pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(
                    neg_prob, neg_labels)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)

            # 多分类
            m_loss = self.m_loss(output_m, labels_m)
            # m_loss = 0
            output_no, output_id = torch.max(output_m, 1)
            correct_m = (output_id == labels_m).sum().data.cpu().numpy()
            # print('labels_m', labels_m.size())
            # print('output_no', output_no.size(), output_no.dtype)
            # print('output_id', output_id.size(), output_id.dtype)
            # print('correct_m', correct_m)

            # print(err)

        else:
            regress_losses = [0, 0, 0, 0]
            classify_loss = 0.5 * self.classify_loss(neg_prob, neg_labels)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0, 0, 0, 0]
        classify_loss_data = classify_loss.data

        loss = classify_loss + m_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)

        return [loss, classify_loss_data ] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total] + [m_loss.data.cpu(), correct_m, len(labels_m) ]


