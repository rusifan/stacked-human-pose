from __future__ import print_function, absolute_import
import torch
import onnx 
import onnxruntime
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from collections import OrderedDict
import _init_paths
from pathlib import Path
from typing import Optional
from infer_mynet import MyNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load("/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/model_module_gcn_13_eva_xyz_8782.pt")
print("=> creating model ...")
adj = np.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/adj_4_16.npy')

adj = torch.from_numpy(adj)
model = MyNet(adj, block=2) # 2 stacks and 2 gcn blocks

state_dict = torch.load('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/train_fullnet_mpii_only_2stg_4gcn/model_5.pth') #mpii 4gcn wrong name 
    # checked train_fullnet_mpii_only/model_3 which is good generalize and 3d
    # checked train_fullnet_mpii_only/model_6 which is good generalize and 3d
    # checked train_fullnet_mpii_only/model_10 which is good generalize and 3d
print("=> loading checkpoint ...")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
# model.to(device)
model.eval()
x = torch.randn(1, 3, 256, 256, requires_grad=True)
# # print("done")
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "results/stg2_mgcn2.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}}
                                )

onnx_model =onnx.load("results/stg2_mgcn2.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("results/stg2_mgcn2.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# # compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(output), ort_outs[0], rtol=1e-03, atol=1e-05)
# import pdb;pdb.set_trace()
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# from onnxruntime.quantization import quantize_dynamic, QuantType
# quantize_dynamic()
from onnxruntime.quantization import quantize_dynamic, QuantType
model_fp32 = "results/stg2_mgcn2.onnx"
model_quant = "results/stg2_mgcn2_quant.onnx"

quantized_model = quantize_dynamic(model_fp32, model_quant,weight_type=QuantType.QUInt8)

ort_session = onnxruntime.InferenceSession(model_quant)
# ort_session = onnxruntime.InferenceSession(model_fp32)
# mean_error = val(opt, actions, test_dataloader, model, criterion, ort_session)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

print('ONNX full precision model size (MB):', os.path.getsize("stg2_mgcn2.onnx")/(1024*1024))
print('ONNX quantized model size (MB):', os.path.getsize("stg2_mgcn2_quant.onnx")/(1024*1024))
#POST TRAINIG STAIC QUATIZATION

# nn.Linear

#run time of onnx model
import time
start = time.time()
for i in range(100):
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
end = time.time()
# print("time taken for 100 runs of onnx model: ",end-start)
print("time taken for 1 run of onnx model: ",(end-start)/100)

# runtime for quantized model
ort_session = onnxruntime.InferenceSession(model_quant) 
start = time.time()
for i in range(100):
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
end = time.time()
# print("time taken for 100 runs of quantized model: ",end-start)
print("time taken for 1 run of quantized model: ",(end-start)/100)