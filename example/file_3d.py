from __future__ import print_function, absolute_import
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


DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]


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

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x
# initialize the camera
# cap = cv2.VideoCapture(0)

# create a figure and an Axes3D object for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loop through the frames
print("=> creating model ...")
adj = np.load('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/clone_of_netscratch/stacked-human-pose/results/adj_4_16.npy')

adj = torch.from_numpy(adj).to(device)
model = MyNet(adj, block=2)

state_dict = torch.load('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/clone_of_netscratch/stacked-human-pose/results/train_fullnet_mpii_only/model_10.pth') #mpii
    # checked train_fullnet_mpii_only/model_3 which is good generalize and 3d
    # checked train_fullnet_mpii_only/model_6 which is good generalize and 3d
    # checked train_fullnet_mpii_only/model_10 which is good generalize and 3d
print("=> loading checkpoint ...")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = torch.nn.DataParallel(model).to(device)

# model_pre = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True).to(device) # two models not
classes = ['person']
print("=> reading image ...")
# read a directory with images
# directory = Path('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/to_send/S1_Directions_1.54138969/')
directory = Path('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/to_send/crop/')
# directory = Path('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/new_images/test_img')
# directory = Path('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/to_send/S9_Discussion_1.54138969/')
# "C:/Users/htaed/Documents/uni_study/thesis_possible/demo/new_images/test_img/yoga.png"

for file in directory.iterdir():
    # print(file)
    # file = Path("C:/Users/htaed/Documents/uni_study/thesis_possible/demo/to_send/crop/S1_Directions_1.54138969_000001.jpg")
    # image_name = cv2.imread(str(file))
    img_2 = cv2.imread(str(file))
    img = cv2.imread(str(file), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    # outputs = model_pre(img)

    # boxes = outputs.pred[0][:, :4]
    # scores = outputs.pred[0][:, 4]
    # class_ids = outputs.pred[0][:, 5].int()

    # # Loop over the results
    # for box, score, class_id in zip(boxes, scores, class_ids):
    #     # Get the class ID and confidence score
    #     class_name = classes[class_id] if class_id < len(classes) else str(class_id)
    #     if class_name == 'person' and score > 0.5:
    #         # Draw a bounding box around the detected person
    #         x1, y1, x2, y2 = box.int().tolist()
    #         # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # # img_2 = cv2.resize(img_2, (256, 256))
    height = img_2.shape[0]
    width = img_2.shape[1]
    # # import pdb;pdb.set_trace()
    # img = img[x1:y1, x2:y2]
    img = cv2.resize(img, (256, 256))
    # img = img.transpose(2, 0, 1)
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    img /= 255
    img = torch.from_numpy(img).float().cuda()
    img = color_normalize(img, DATASET_MEANS, DATASET_STDS)
    img = img.unsqueeze(0)

    left_top = [0, 0]
    ratio_x = 1.0
    ratio_y = 1.0
    with torch.no_grad():
        out_3d, heatmaps = model(img, left_top, ratio_x, ratio_y)
        output = heatmaps[-1]
        # output = output.cpu().numpy()
        # import pdb;pdb.set_trace()
        output = output.unsqueeze(4)
        output = output.detach()
        key_points = soft_argmax(output)
        key_points = key_points.squeeze().detach().cpu().numpy()
        key_points[:, :1] *= height
        key_points[:, 1:2] *= width
        # import pdb;pdb.set_trace()
        
        # plt.imshow(output.sum(axis=0))
        # plt.savefig('C:/Users/htaed/Documents/uni_study/thesis_possible/demo/clone_of_netscratch/stacked-human-pose/results/heatmap_test_my.png')
        # plt.show()
        # plt.clf()
        # kps_3d [1:] += kps_3d [:1]
        # print(kps_3d)
        # print("pred#"*100)
        out_3d = out_3d.squeeze()
        out_3d = out_3d.permute(1, 0)
        # import pdb;pdb.set_trace()
        out_3d = out_3d.detach().cpu().numpy()
        out_3d [1:] += out_3d [:1]
        keypoints = out_3d
    skeleton = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9],
    [7, 10], [10, 11], [11, 12],
    [7, 13], [13, 14], [14, 15]
    ] 

    # clear the previous plot
    ax.cla()

    # plot the 3D keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2])
    for connection in skeleton:
        x = [keypoints[connection[0], 0], keypoints[connection[1], 0]]
        y = [keypoints[connection[0], 1], keypoints[connection[1], 1]]
        z = [keypoints[connection[0], 2], keypoints[connection[1], 2]]
        ax.plot(x, y, z, c='b')
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([4, 6])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=-79, azim=-90)
    # show the camera feed and the plot in one window
    # cv2.imshow('Camera Feed', img_2)
    plt.pause(0.001)
    plt.draw()
    for pts in key_points:
        cv2.circle(img_2, (int(pts[0]), int(pts[1])), 3, (0, 0, 255), -1)
        cv2.imshow('image', img_2)
    # press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close all windows
cv2.destroyAllWindows()
