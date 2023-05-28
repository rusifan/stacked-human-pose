from __future__ import print_function, absolute_import
import _init_paths

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from infer_mynet import MyNet
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj = np.load('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/clone_of_netscratch/stacked-human-pose/results/adj_4_16.npy')
adj = torch.from_numpy(adj).to(device)
model = MyNet(adj,num_stacks =2, block=2).to(device) # 2 stacks and 2 gcn blocks

model.eval()
left_top = [0, 0]
ratio_x = 1.0
ratio_y = 1.0
example = torch.rand(1, 3, 256, 256).to(device)
traced_script_module = torch.jit.trace(model, example, strict=False)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("C:/Users/htaed/Documents/uni_study/thesis_possible/demo/clone_of_netscratch/stacked-human-pose/for_mobile/model_notStrict.ptl")
