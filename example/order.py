from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets
import pose.losses as losses
import wandb
from pose.datasets.human36_dataloader import hum36m_dataloader
from tqdm import tqdm
from pose.models.hourglass import *
import torch.nn as nn
import numpy as np
from model.modulated_gcn import ModulatedGCN

def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)
    #shape of kps_mask (1002, 1000, 3)
    import pdb;pdb.set_trace()
    # Draw the keypoints.
    for l in range(len(kps_lines)):
        # import pdb;pdb.set_trace()
        i1 = kps_lines[l][0] #0 for l = 0
        i2 = kps_lines[l][1] #7 for l = 0 check skeliton shape tupple   kps shape [3,17]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            # import pdb;pdb.set_trace()
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

def cam2pixel(cam_coord, f, c): #cam_coord shape (17,3)
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    print(f'image_coord shape {img_coord.shape}')
    return img_coord



def soft_argmax(voxels):
	"""
	Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
	Return: 3D coordinates in shape (batch_size, channel, 3)
	"""
	assert voxels.dim()==5
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 1000.0 
	N,C,H,W,D = voxels.shape
	soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
	soft_max = soft_max.view(voxels.shape)
	indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0).to('cuda')
	indices_kernel = indices_kernel.view((H,W,D))
	conv = soft_max*indices_kernel
	indices = conv.sum(2).sum(2).sum(2)
	z = indices%D
	y = (indices/D).floor()%W
	x = (((indices/D).floor())/W).floor()%H
	coords = torch.stack([y,x,z],dim=2)
	return coords[:,:,:2]/64.0


# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))


# init global variables
best_loss = 10000
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

def main():
    global best_loss
    global idx
    train_batch = 16
    test_batch = 16
    workers = 1
    epochs = 100
    schedule = [60,90]
    gamma = 0.1
    sigma_decay = 0
    debug = False
    flip = False
    wandb_flag = False
    annotation_path_train = "annotation_body3d/fps25/h36m_train.npz"
    annotation_path_test = "annotation_body3d/fps25/h36m_test.npz"
    root_data_path = "/netscratch/nafis/human-pose/dataset_human36_nos7_f25"
    # create model
    # model = hg(num_stacks=4, num_blocks=1, num_classes=17)
    model = hg(num_stacks=4, num_blocks=1, num_classes=16)
    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)


    # load mgcn model
    adj = np.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/adj_4_16.npy')
    adj = torch.from_numpy(adj).float().to(device)
    mgcn = ModulatedGCN(adj, 384, num_layers=2, p_dropout=0, nodes_group=None).to('cuda')

    # define loss function (criterion) and optimizer
    # criterion = losses.JointsMSELoss().to(device)

    train_dataset = hum36m_dataloader(root_data_path, annotation_path_train, True, [1.1, 2.0], False, 5, flip_prob = 1)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=train_batch, shuffle=True,
    #     num_workers=workers, pin_memory=True
    # )

    # val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_dataset = hum36m_dataloader(root_data_path, annotation_path_test, True, [1.1, 2.0], False, 5, flip_prob = 1)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=test_batch, shuffle=False,
    #     num_workers=workers, pin_memory=True
    # )
    # checkpoint = torch.load('/netscratch/nafis/human-pose/pytorch-pose/results/human36/model_6.pth')
    r = train_dataset.__getitem__(0)
    kps = r['kp_2d_org']
    image_path = r['image_name']
    filename = '/netscratch/nafis/human-pose/pytorch-pose/results/img/human_order/'
    img = cv2.imread(image_path)
    for i in range(16):
        tmp = cv2.circle(cv2.imread(image_path), (int(kps[i][0]), int(kps[i][1])), 4, (0, 0, 255), -1)
        cv2.imwrite(filename + f'order{i}.jpg', tmp)
if __name__ == '__main__':
    main()
