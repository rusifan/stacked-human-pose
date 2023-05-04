from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *


# def map_mpii_to_common(kps_mpii):
#     kps_human36m = [0] * 14
#     kps_human36m[0] = kps_mpii[6]   # pelvis
#     kps_human36m[1] = kps_mpii[2]   # right hip
#     kps_human36m[2] = kps_mpii[1]   # right knee
#     kps_human36m[3] = kps_mpii[0]   # right ankle/foot
#     # kps_human36m[4] = kps_mpii[4]   # left hip
#     kps_human36m[4] = kps_mpii[4]   # left knee
#     kps_human36m[5] = kps_mpii[5]   # left ankle/foot
#     # kps_human36m[7] = kps_mpii[7]   # spine
#     kps_human36m[6] = kps_mpii[7]   # thorax
#     # kps_human36m[9] = kps_mpii[9]   # head top
#     kps_human36m[7] = kps_mpii[9] # head top
#     kps_human36m[8] = kps_mpii[13] # left_shoulder
#     kps_human36m[9] = kps_mpii[14] # right wrist
#     kps_human36m[10] = kps_mpii[15] # left_wrist
#     kps_human36m[11] = kps_mpii[12] # right_shoulder
#     kps_human36m[12] = kps_mpii[11] # right_elbow
#     kps_human36m[13] = kps_mpii[10] # right_wrist
    
#     #return the list of 14 keypoints as tensor of shape (14,3)
#     return torch.stack(kps_human36m)
    # return np.array(kps_human36m).reshape((14,3))
    # return torch.Tensor(kps_human36m)


def map_mpii_to_common(kps_mpii):
    kps_human36m = [0] * 16
    kps_human36m[0] = kps_mpii[6]   # pelvis
    kps_human36m[1] = kps_mpii[2]   # right hip
    kps_human36m[2] = kps_mpii[1]   # right knee
    kps_human36m[3] = kps_mpii[0]   # right ankle/foot
    kps_human36m[4] = kps_mpii[3]   # left hip
    kps_human36m[5] = kps_mpii[4]   # left knee
    kps_human36m[6] = kps_mpii[5]   # left ankle/foot
    # kps_human36m[7] = kps_mpii[7]   # spine
    kps_human36m[7] = kps_mpii[7]   # thorax
    kps_human36m[8] = kps_mpii[8]   # uppper neck
    kps_human36m[9] = kps_mpii[9] # head top
    kps_human36m[10] = kps_mpii[13] # left_shoulder
    kps_human36m[11] = kps_mpii[14] # left elbow
    kps_human36m[12] = kps_mpii[15] # left_wrist
    kps_human36m[13] = kps_mpii[12] # right_shoulder
    kps_human36m[14] = kps_mpii[11] # right_elbow
    kps_human36m[15] = kps_mpii[10] # right_wrist
    
    return torch.stack(kps_human36m)
# def map_mpii_to_common(kps_mpii): #wrong
#     kps_human36m = [0] * 16
#     kps_human36m[0] = kps_mpii[6]   # pelvis
#     kps_human36m[1] = kps_mpii[3]   # right hip
#     kps_human36m[2] = kps_mpii[4]   # right knee
#     kps_human36m[3] = kps_mpii[5]   # right ankle/foot
#     kps_human36m[4] = kps_mpii[2]   # left hip
#     kps_human36m[5] = kps_mpii[1]   # left knee
#     kps_human36m[6] = kps_mpii[0]   # left ankle/foot
#     # kps_human36m[7] = kps_mpii[7]   # spine
#     kps_human36m[7] = kps_mpii[7]   # thorax
#     kps_human36m[8] = kps_mpii[8]   # uppper neck
#     kps_human36m[9] = kps_mpii[9] # head top
#     kps_human36m[10] = kps_mpii[12] # left_shoulder
#     kps_human36m[11] = kps_mpii[11] # left elbow
#     kps_human36m[12] = kps_mpii[10] # left_wrist
#     kps_human36m[13] = kps_mpii[13] # right_shoulder
#     kps_human36m[14] = kps_mpii[14] # right_elbow
#     kps_human36m[15] = kps_mpii[15] # right_wrist
    
#     return torch.stack(kps_human36m)


class Mpii(data.Dataset):
    def __init__(self, is_train = True, **kwargs):
        self.img_folder = kwargs['image_path'] # root image folders
        self.jsonfile   = kwargs['anno_path']
        self.is_train   = is_train # training set or test set
        self.inp_res    = kwargs['inp_res']
        self.out_res    = kwargs['out_res']
        self.sigma      = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']

        # import pdb;pdb.set_trace()
        # create train/val split
        with open(self.jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train_list, self.valid_list = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid_list.append(idx)
            else:
                self.train_list.append(idx)
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        meanstd_file = './data/mpii/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train_list:
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train_list)
            std /= len(self.train_list)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train_list[index]]
        else:
            a = self.anno[self.valid_list[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['joint_self'])
        # to convert to common format 2/4/2023
        pts = map_mpii_to_common(pts)
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        # import pdb;pdb.set_trace()
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            # r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            r = 0

            # Flip
            # if random.random() <= 0.5:
            #     img = torch.from_numpy(fliplr(img.numpy())).float()
            #     pts = shufflelr(pts, width=img.size(2), dataset='mpii')
            #     c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            # import pdb;pdb.set_trace()
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta info
        left_top = [0.0, 0,0]
        ratio_x = 1
        ratio_y = 1
        meta = {'index' : index, 'center' : c, 'scale' : s, 'img_path':img_path,
        'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight,
        'left_top': left_top, 'ratio_x': ratio_x, 'ratio_y': ratio_y} #newly added for making the network work


        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)


def mpii(**kwargs):
    return Mpii(**kwargs)

mpii.njoints = 16  # ugly but works
# mpii.njoints = 14  # ugly but works
