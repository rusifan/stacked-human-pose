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
from mynet import MyNet
# from mynet_sgcn import MyNet #sgcn
# from net_graformer import MyNet #graformer
# from mynet_8hg_4mgcn import MyNet #8hg_4mgcn
# from mynet_2hg_2mgcn import MyNet #2hg_2mgcn
# from net_without_softmax import MyNet #16kps

print("=> creating model ...")
adj = np.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/adj_4_16.npy')
adj = torch.from_numpy(adj).to('cuda')
block = 2
num_stacks = 4
print("block and num_stacks")
print(block, num_stacks)
# model = MyNet(adj, block=2, num_stacks=2) # 2stack and 2 gcn
# model = MyNet(adj, block=2) # 4stack and 2 sem-gcn --semgcn
# model = MyNet(adj, block=2, num_stacks=8) # 8stack and 2 gcn
model = MyNet(adj, block=block, num_stacks=num_stacks) # 8stack and 2 gcn
# model = MyNet(adj, block=4, num_stacks=4) # 8stack and 4 gcn
# model = torch.nn.DataParallel(model).cuda()
model = model.cuda()

input = torch.randn(1, 3, 256, 256).cuda()
left_top=[[0, 0],[0,0]]
ratio_x=[1.0]
ratio_y=[1.0]
torch.cuda.synchronize()
t1 = time.time()
for i in range(100):
    output = model(input, left_top, ratio_x, ratio_y)
torch.cuda.synchronize()
t2 = time.time()
# print("for graformer")
print('time:', (t2 - t1)/100)
print('fps', 1/((t2 - t1)/100))

#number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters())
# print("number of parameters:", pytorch_total_params)
print("number of parameters:", pytorch_total_params/1000000, "million")

