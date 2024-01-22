import torch
import torch.nn as nn
import _init_paths
import pose.models as models
from model.modulated_gcn import ModulatedGCN
from pose.models.hourglass import *
from model.graphunet import GraphNet



def get_state_dict():
    print("loading state dict")
    # state_dict = torch.load('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/stacked2_16kps_fix/model_61.pth') #16kps checkpoint pre trained on mpii
    state_dict = torch.load('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/human36_gaus_1/model_2.pth') #16kps checkpoint pre trained on mpii
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
    # import pdb;pdb.set_trace()
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class MyNet(nn.Module):
    def __init__(self, adj, block, num_stacks) -> None:
        super(MyNet, self).__init__()
        print("net without softmax")
        self.sthg = hg(num_stacks=num_stacks, num_blocks=1, num_classes=16)
        # self.sthg = hg(num_stacks=4, num_blocks=1, num_classes=16)
        self.sthg.load_state_dict(get_state_dict())
        self.graphnet = GraphNet(in_features=64*64, out_features=2)
        self.MGCN = ModulatedGCN(adj, 384, num_layers=block, p_dropout=0, nodes_group=None)
        # self.MGCN.load_state_dict(torch.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results_2layers/model_module_gcn_20_eva_xyz_5236.pth'))

    def forward(self, x ,left_top, ratio_x, ratio_y):
        N = x.shape[0]
        heatmaps = self.sthg(x)
        output = heatmaps[-1]
        output = output.view(N, 16, -1)
        # key_points = self.graphnet(output)
        key_points = self.graphnet(output)
        # import pdb;pdb.set_trace()
        key_points = key_points.view(N, -1, 16, 2, 1).permute(0, 3, 1, 2, 4)
        out_3d = self.MGCN(key_points)

        return out_3d, heatmaps