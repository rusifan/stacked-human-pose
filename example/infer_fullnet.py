from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import os
import time
from collections import OrderedDict
import _init_paths
import tqdm
import tqdm
import wandb
import matplotlib.pyplot as plt
import cv2

import numpy as np
import pose.losses as losses
import pose.models as models
import pose.datasets as datasets
from pose.datasets.human36_dataloader import hum36m_dataloader #16kps
from pose.utils.evaluation import accuracy, AverageMeter
from pose.utils.imutils import batch_with_heatmap
from pose.utils.logger import Logger, savefig
from pose.utils.misc import save_pred, adjust_learning_rate
from pose.utils.osutils import isfile, join
from pose.utils.transforms import fliplr, flip_back
from pose.utils.visualize import show, show_heatmap 
from infer_mynet import MyNet

DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def cam2pixel(cam_coord, f, c): #cam_coord shape (17,3)
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    print(f'image_coord shape {img_coord.shape}')
    return img_coord



def main():
    annotation_path_train = "annotation_body3d/fps25/h36m_train.npz"
    annotation_path_test = "annotation_body3d/fps25/h36m_test.npz"
    root_data_path = "/netscratch/nafis/human-pose/dataset_human36_nos7_f25"

    print("=> creating model ...")
    adj = np.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/adj_4_16.npy')
    adj = torch.from_numpy(adj).to('cuda')
    model = MyNet(adj, block=2)

    #load pretrained model
    # state_dict = torch.load('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/full_net16kps/model_10.pth') #16kps checkpoint pre trained on mpii
    state_dict = torch.load('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/train_fullnet_mpii_only/model_10.pth') #mpii
    # checked train_fullnet_mpii_only/model_3 which is good generalize and 3d
    # checked train_fullnet_mpii_only/model_6 which is good generalize and 3d
    # checked train_fullnet_mpii_only/model_10 which is good generalize and 3d
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model).cuda()
    val_dataset = hum36m_dataloader(root_data_path, annotation_path_test, True, [1.1, 2.0], False, 5, flip_prob=1)

    # Data loading code
    print("=> loading data ...")
    train_dataset = hum36m_dataloader(root_data_path, annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob=1)
    model.eval()
    data = train_dataset.__getitem__(10)
    img = data['image'].unsqueeze(0).cuda()
    kps_2d = data['kp_2d'].cuda()
    kps_3d = data['kp_3d'].cuda()
    heatmap = data['heatmap'].cuda()
    ratio_x = data['ratio_x']
    ratio_y = data['ratio_y']
    left_top = data['leftTop']
    image_name = data['image_name']

    # fgor test images
    image_name = '/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/test_img/right_hand_crop.jpg' #for test_img
    img = cv2.imread(image_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    img = cv2.resize(img, (256, 256))
    # img = img.transpose(2, 0, 1)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img /= 255
    img = torch.from_numpy(img).float().cuda()
    img = color_normalize(img, DATASET_MEANS, DATASET_STDS)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        out_3d, heatmaps = model(img, left_top, ratio_x, ratio_y)
        output = heatmaps[-1]
        # output = output.cpu().numpy()
        # import pdb;pdb.set_trace()
        output = output.squeeze().detach().cpu().numpy()
        plt.imshow(output.sum(axis=0))
        plt.savefig('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/vis/heatmap_test_my.png')
        plt.clf()
        kps_3d [1:] += kps_3d [:1]
        print(kps_3d)
        print("pred#"*100)
        out_3d = out_3d.squeeze()
        out_3d = out_3d.permute(1, 0)
        # import pdb;pdb.set_trace()
        out_3d = out_3d.detach().cpu().numpy()
        out_3d [1:] += out_3d [:1]
        print(out_3d)

        cvimg = cv2.imread(image_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        c = np.array([[512.54150496],[515.45148698]])
        f = np.array([[1145.04940459],[1143.78109572]]) #for camera 54138969
        # c = np.array([[1.33006922e+03],[2.17054342e+03]])
        # f = np.array([[3.35396544e+03],[3.35360455e+03]]) #for my camera 
        pred = cam2pixel(out_3d, f, c)

        for i in pred:
            cv2.circle(cvimg, (int(i[0]), int(i[1])), 3, (0,0,255), -1)
        cv2.imwrite('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/vis/pred_3dkps_test_my.png', cvimg)
if __name__ == '__main__':
    main()