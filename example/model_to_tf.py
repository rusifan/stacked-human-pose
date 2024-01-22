from __future__ import print_function, absolute_import
import torch
import onnx 
import onnxruntime
# import numpy as np
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import os
# from torch.utils.data import DataLoader
# import torch.quantization
# from torch.quantization import QuantStub, DeQuantStub
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

import tensorflow as tf
from pathlib import Path
import onnx 
from onnx_tf.backend import prepare
# import onnx_tf.backend.prepare as prepare

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
# model = torch.load("/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/model_module_gcn_13_eva_xyz_8782.pt")
# print("=> creating model ...")
# # adj = np.load('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/clone_of_netscratch/stacked-human-pose/results/adj_4_16.npy')
# adj = np.load('/netscratch/nafis/human-pose/Modulated-GCN/Modulated_GCN/Modulated-GCN_benchmark/results/adj_4_16.npy')

# adj = torch.from_numpy(adj).to(device)
# model = MyNet(adj, block=2) # 2 stacks and 2 gcn blocks

# state_dict = torch.load('/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/results/train_fullnet_mpii_only_2stg_4gcn/model_5.pth') #mpii 4gcn wrong name 
# # state_dict = torch.load('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/clone_of_netscratch/stacked-human-pose/train_fullnet_mpii_only_2stg_2gcn/model_5.pth') #mpii
#     # checked train_fullnet_mpii_only/model_3 which is good generalize and 3d
#     # checked train_fullnet_mpii_only/model_6 which is good generalize and 3d
#     # checked train_fullnet_mpii_only/model_10 which is good generalize and 3d
# print("=> loading checkpoint ...")
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# model.load_state_dict(new_state_dict)
# model.to(device)
# model.eval()
# x = torch.randn(1, 3, 256, 256, requires_grad=True).to(device)
# # # print("done")
# torch.onnx.export(model,               # model being run
#                 #   (x, [0, 0], 1.0, 1.0),  #for multi                       # model input (or a tuple for multiple inputs)
#                   x,                         # model input (or a tuple for multiple inputs)
#                   "stg2_mgcn2_fortf.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=12,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                 #   input_names = ['input', 'top_left', 'x_scale', 'y_scale'],   # the model's input names
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {1 : 'batch_size'},    # variable length axes
#                                 'output' : {1 : 'batch_size'}}
#                                 )

onnx_model =onnx.load("/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/example/stg2_mgcn2_fortf.onnx")
onnx.checker.check_model(onnx_model)
# from onnxruntime.quantization import quantize_dynamic, QuantType
# quantize_dynamic()

#onnx to tf 


onnx_model = onnx.load("/netscratch/nafis/human-pose/new_code_to_git/stacked-human-pose/example/stg2_mgcn2_fortf.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("stg2_mgcn2_fortf.pb")

print("Model is successfully converted from ONNX to TF")


tf_model = Path("stg2_mgcn2_fortf.pb")
# onnx_model = onnx.load("stg2_mgcn2_fortf.onnx")

#tf light path
tf_lite_path = "stg2_mgcn2_fortf.tflite"

#convert onnx to tf lite
converter = tf.lite.TFLiteConverter.from_frozen_graph(onnx_model)
tflite_model = converter.convert()

#save tf lite model
with open(tf_lite_path, "wb") as f:
    f.write(tflite_model)

#find the model size of tf lite model
tf_lite_model_size = Path(tf_lite_path).stat().st_size
print("TF Lite model is %d bytes" % tf_lite_model_size)
x = torch.randn(1, 3, 256, 256, requires_grad=True)

#pass a random input to the model
input_data = np.array(np.random.random_sample(x.shape), dtype=np.float32)
# input_data = np.array(np.random.random_sample((1, 3, 256, 256)), dtype=np.float32)
# start the session and pass the input
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# calculate the runtime for tf lite model
import time
start_time = time.time()
interpreter.invoke()
print("--- %s seconds ---" % (time.time() - start_time))


